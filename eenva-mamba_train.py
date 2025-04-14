from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.')
# ROOT = ''
print(ROOT)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.normpath(os.path.join(ROOT,
                                                                                  'ultralytics/cfg/datasets/yourdataset.yaml')),
                        help='dataset.yaml path')
    parser.add_argument(
        '--config', type=str,
        default=os.path.normpath(os.path.join(ROOT, 'ultralytics/cfg/models/eenvamamba/EEnvA_Mamba.yaml')),
        help='model path(s)')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size',
                        type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train',
                        help='train, val, test, speed or study')
    # parser.add_argument('--device', default='0,1,2,3,4,5,6,7',
    #                     help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=1,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=os.path.normpath(
        os.path.join(ROOT, 'output_dir/mscoco')), help='save to project/name')
    parser.add_argument('--name', default='eenvamamba',
                        help='save to project/name')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    print(ROOT)

    opt = parse_opt()
    task = opt.task

    print(opt.data)

    args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": opt.name,
    }
    
    for key, value in args.items():
        print(key, value)

    model_conf = opt.config
    task_type = {
        "train": YOLO(model_conf).train(**args),
        "val": YOLO(model_conf).val(**args),
        # "test": YOLO(model_conf).test(**args),
    }
    task_type.get(task)
