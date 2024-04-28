from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count

def main(args):
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    input_torch = torch.randn((1,3,640,640))
    flops = FlopCountAnalysis(model, input_torch)
    print("Total Flops:",flops.total())
    print("Total Flops:",flop_count(model,input_torch))
    
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )