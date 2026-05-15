import torch
import os
from enum import Enum
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from typing import Optional
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers

from internvl.model.var_vae.models import VQVAE, build_vae
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from internvl.model.internvl_chat import InternVLGenSeg
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from internvl.train.dataset import build_transform
import itertools
from pycocotools import mask


def vis_multi_mask(img_path, multi_masks, idx = 0):
    import cv2
    img = cv2.imread(img_path)[:,:,::-1]
    def vis_mask(img, mask, alpha=0.5):
        mask_resized = cv2.resize(mask, img.shape[:2][::-1])
        mask_binary = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)
        overlay = np.zeros_like(img)
        overlay[binary > 0] = (0, 255, 0)  # 默认绿色
        result = cv2.addWeighted(img, 1, overlay, alpha, 0)
        return Image.fromarray(result)

    # Define the number of images per row
    images_per_row = 4
    total_images = 8
    rows = (total_images + images_per_row - 1) // images_per_row
    image_width, image_height = img.shape[:2][::-1]
    stitched_image = Image.new('RGB', (images_per_row * image_width, rows * image_height))

    # Paste the images into the new image
    stitched_image.paste(Image.fromarray(img), (0 * image_width, 0 * image_height))
    for i in [3,4,5,6,7,8,9]:
        cur_img = vis_mask(img, multi_masks[i])
        row = (i-2) // images_per_row
        col = (i-2) % images_per_row
        stitched_image.paste(cur_img, (col * image_width, row * image_height))

    # Save the stitched image
    output_path = f"results/multi_masks/{idx}-multi_mask.jpg"
    stitched_image.save(output_path) 


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/path/to/val2017')
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'pad'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/path/to/coco'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/panoptic_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    gen_img_reso: int = 256
    dynamic_image_size: int = 1
    pad2square: bool = False


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, json_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        print(f"Evaluating dataset from path: {json_path}")
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        if "grefcoco" in json_path:
            self.data = [item for item in self.data if len(item['instruction'])>0]
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.tokenizer = tokenizer
        print(f'[Dataset] dynamic_image_size: {self.data_args.dynamic_image_size}')
        print(f'[Dataset] pad2square: {self.data_args.pad2square}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            dynamic = False if max_num == 1 else True
            if dynamic:
                images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                transform = build_transform(False, input_size=input_size, pad2square=False)
            else:
                # [image.size for image in images]
                images = [image]
                transform = build_transform(False, input_size=input_size, pad2square=self.data_args.pad2square)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values
        data = self.data[index]
        image_file = data['image_info']['file_name']
        image_folder = self.data_args.refcoco_image_folder
        img_path = os.path.join(image_folder, image_file)
        # print(img_path)
        image = Image.open(img_path).convert("RGB")
        H, W, _ = np.array(image).shape

        instruction = ''
        for sent in data['instruction']:
            instruction += ' {}.'.format(sent['sent'])

        image_info = data['image_info']
        m_final = np.zeros(
                (image_info["height"], image_info["width"])
            ).astype(np.uint8)
        for ann in data['anns']:
            if len(ann["segmentation"]) == 0:
                m = np.zeros(
                    (image_info["height"], image_info["width"])
                ).astype(np.uint8)
            else:
                if type(ann["segmentation"]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"], image_info["height"], image_info["width"], )
                else:
                    rle = ann["segmentation"]
                    if isinstance(rle["counts"], list):
                        rle = mask.frPyObjects(
                            [rle], image_info["height"], image_info["width"]
                        )
                    elif not isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
            m_final = m_final | m
        seg_mask = m_final
        seg_mask = seg_mask[:,:, None]

        image = load_image(img_path, max_num=self.data_args.dynamic_image_size)
        inp = f"<image>\n Given the following instructions: {instruction}; please perform referring segmentation on this image."

        # return image, seg_mask, inp
        return {
            'image': image,
            'seg_mask': seg_mask,
            'text': inp,
        }
    
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def parse_outputs(outputs,gt_mask):
    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        scores = output['instances'].scores.cpu().numpy()
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        res = {
            'pred':pred_mask,
            'gt': gt_mask,
            'scores':scores,
            'pred_cls':pred_cls
        }
        res_list.append(res)
    return res_list

def compute_metric(intersection_meter,union_meter,acc_iou_meter, gt_cls, results_list):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    for results in results_list:
        gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        preds = preds.astype(np.uint8)
        # pick mask with maximum score
        topk_scores,idx = torch.topk(torch.tensor(scores),1)
        idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        if results['pred_cls'] is not None:
            topk_pred_cls = results['pred_cls'][idx]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        # here topk=1, len(topk_preds)=1
        for i,pred_ in enumerate(topk_preds):
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous().clone(), torch.tensor(gt).int().cuda().contiguous(), 2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        gt_list.append(gt)

    return pred_list,gt_list

def get_pre_mask_array(H, W, gen_img_reso, pre_mask):
    aspect_ratio = W / H    
    pre_mask = pre_mask.squeeze().clone().add_(1).mul_(0.5)  # Remove batch dimension if present
    if len(pre_mask.shape) == 3:  # If RGB (3, H, W)
        pre_mask = pre_mask.mean(dim=0)  # Average across channels to get grayscale
    
    if aspect_ratio > 1:  # width > height
        new_h = int(gen_img_reso / aspect_ratio)
        padding_y = (gen_img_reso - new_h) // 2
        # Remove padding and resize
        pre_mask = pre_mask[padding_y:padding_y + new_h, :]
        pre_mask = torch.nn.functional.interpolate(
            pre_mask.unsqueeze(0).unsqueeze(0), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
    else:  # height >= width
        new_w = int(gen_img_reso * aspect_ratio)
        padding_x = (gen_img_reso - new_w) // 2
        # Remove padding and resize
        pre_mask = pre_mask[:, padding_x:padding_x + new_w]
        pre_mask = torch.nn.functional.interpolate(
            pre_mask.unsqueeze(0).unsqueeze(0), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
    return pre_mask

def get_pre_mask_PIL(H, W, gen_img_reso, pre_mask):
    aspect_ratio = W / H
    if aspect_ratio > 1:  # width > height
        new_h = int(gen_img_reso / aspect_ratio)
        padding_y = (gen_img_reso - new_h) // 2
        # Remove padding and resize
        pre_mask = pre_mask.crop((0, padding_y, gen_img_reso, padding_y + new_h))
        pre_mask = pre_mask.resize((W, H), Image.BILINEAR)
    else:  # height >= width
        new_w = int(gen_img_reso * aspect_ratio)
        padding_x = (gen_img_reso - new_w) // 2
        # Remove padding and resize
        pre_mask = pre_mask.crop((padding_x, 0, padding_x + new_w, gen_img_reso))
        pre_mask = pre_mask.resize((W, H), Image.BILINEAR)

    pre_mask = torch.tensor(np.array(pre_mask.convert('L'))/255.0)
    return pre_mask

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def collate_fn(batches, tokenizer):
    images = [_['image'] for _ in batches]
    seg_masks = [_['seg_mask'] for _ in batches]
    texts = [_['text'] for _ in batches]
    return images, seg_masks, texts

def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]


    model_path = data_args.model_path
    print(f'current model is {model_path}')
    InternVLGenSeg.gen_img_reso = data_args.gen_img_reso
    model = InternVLGenSeg.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True)
    print("Model Loaded")
    model = model.eval().cuda()
    print("Model to cuda done")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    gen_img_reso = data_args.gen_img_reso
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16) if gen_img_reso == 256 else (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    vae = build_vae(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device="cuda",
        patch_nums = patch_nums,
    )
    model.set_visual_tokenizer(vae)

    data_args.refcoco_image_folder = data_args.image_folder
    eval_dataset = CustomDataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)


    # Initialize progress bar
    running_ciou = 0.0
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    cur_id = 0

    from functools import partial
    dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            sampler=InferenceSampler(len(eval_dataset)),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )
    
    outputs = []
    pbar = tqdm(dataloader, desc='Evaluating')
    idx = 0
    for images, seg_masks, questions in pbar:
        # data_in = eval_dataset.__getitem__(i)
        # image, seg_mask, question = data_in
        image, seg_mask, question = images[0].to(torch.bfloat16).cuda(), seg_masks[0], questions[0]
        image = image.unsqueeze(0)
        model.image_num = 0

        model.language_model.gen_cache = []
        response = model.chat(tokenizer, image[0], question, generation_config)

        # outputs = tokenizer.decode(output_ids[0]).strip()
        gt_mask = seg_mask*255
        pre_mask = model.language_model.gen_cache[0]
        H, W, _ = gt_mask.shape

        # Calculate padding based on original aspect ratio
        if data_args.pad2square:
            pre_mask = get_pre_mask_PIL(H, W, gen_img_reso, pre_mask)
        else:
            pre_mask = pre_mask.resize((W, H), Image.BILINEAR)
            # pre_mask.save(f'a-{i}-pre.png')
            pre_mask = torch.tensor(np.array(pre_mask.convert('L'))/255.0)

        masks_list = torch.tensor(gt_mask[:,:,0] > 127).cuda(0).int()
        output_list = (pre_mask > 0.5).cuda(0).int()
        
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        intersection_i, union_i, _ = intersectionAndUnionGPU(masks_list.contiguous().clone(), output_list.contiguous(), 2, ignore_index=255)
        intersection += intersection_i
        union += union_i
        acc_iou += intersection_i / (union_i + 1e-10)
        acc_iou[union_i == 0] = 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / 1

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)
        outputs.append({
            'intersection': intersection,
            'union': union,
        })

        # Calculate and update running ciou
        running_iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        running_ciou = running_iou_class[1]
        pbar.set_postfix({'ciou': f'{running_ciou:.3f}'})
        idx += 1
    
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    if torch.distributed.get_rank() == 0:
        intersection_sum = sum([item['intersection'] for item in merged_outputs])
        union_sum = sum([item['union'] for item in merged_outputs])
        iou_class = intersection_sum / (union_sum + 1e-10)
        ciou = iou_class[1]
        print("ciou {:.3f}".format(ciou))
        giou = acc_iou_meter.avg[1]
        print("giou {:.3f}".format(giou))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    # print("giou {:.3f}; ciou {:.3f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    evaluation()