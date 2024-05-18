MODEL:
  META_ARCHITECTURE: "OVSegDEMO"
  BACKBONE:
    FREEZE_AT: 0
    NAME: "D2SwinTransformerV2"
  SWIN:
    EMBED_DIM: 128
    DEPTHS: [2, 2, 18, 2]
    NUM_HEADS: [4,8,16,32]
    WINDOW_SIZE: 12
    APE: False
    DROP_PATH_RATE: 0.2
    PATCH_NORM: True
    PRETRAIN_IMG_SIZE: 384
    USE_CHECKPOINT: False
    PRETRAINED_WINDOW_SIZES: [ 12, 12, 12, 6 ]
  WEIGHTS: "swinv2_base_patch4_window12_384_22k.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  SEM_SEG_HEAD:
    NAME: "OpenVocabMaskFormerHead"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    IGNORE_VALUE: 255
    NUM_CLASSES: 171 # number of categories in training set
    EMBEDDING_DIM: 768
    EMBED_LAYERS: 2
    COMMON_STRIDE: 4 # not used, hard-coded
    LOSS_WEIGHT: 1.0
    CONVS_DIM: 256
    MASK_DIM: 256
    NORM: "GN"
    # pixel decoder
    PIXEL_DECODER_NAME: "MSDeformAttnPixelDecoder"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: ["res3", "res4", "res5"]
    TRANSFORMER_ENC_LAYERS: 6
  MASK_FORMER:
    TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
    TRANSFORMER_IN_FEATURE: "multi_scale_pixel_decoder"
    DEEP_SUPERVISION: True
    NO_OBJECT_WEIGHT: 0.1
    CLASS_WEIGHT: 2.0
    DICE_WEIGHT: 1.0
    MASK_WEIGHT: 20.0
    HIDDEN_DIM: 256
    NUM_OBJECT_QUERIES: 100
    NHEADS: 8
    DROPOUT: 0.1
    DIM_FEEDFORWARD: 2048
    ENC_LAYERS: 0
    DEC_LAYERS: 6
    PRE_NORM: False
    ENFORCE_INPUT_PROJ: False
    SIZE_DIVISIBILITY: 32
    DEC_LAYERS: 10  # 9 decoder layers, add one for the loss on learnable query
    TRAIN_NUM_POINTS: 12544
    OVERSAMPLE_RATIO: 3.0
    IMPORTANCE_SAMPLE_RATIO: 0.75
    TEST:
      SEMANTIC_ON: True
      INSTANCE_ON: False
      PANOPTIC_ON: False
      OVERLAP_THRESHOLD: 0.8
      OBJECT_MASK_THRESHOLD: 0.8
  CLIP_ADAPTER:
    TEXT_TEMPLATES: "vild"
    CLIP_MODEL_NAME: "ViT-L/14"
    MASK_FILL: "mean"
    MASK_EXPAND_RATIO: 1.0
    MASK_THR: 0.35 # choose the foreground objects
    MASK_MATTING: False # use soft background, default not used
    MASK_PROMPT_DEPTH: 3
    MASK_PROMPT_FWD: True # use mask prompt during forward
    REGION_RESIZED: True # resize to the input of clip, e.g., 224
    CLIP_ENSEMBLE: True # use ensemble of two classification branches
    CLIP_ENSEMBLE_WEIGHT: 0.7
DATASETS:
  TRAIN: ("coco_2017_train_stuff_sem_seg",)
  TEST: ("ade20k_sem_seg_val",)
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.0001
  MAX_ITER: 120000
  WARMUP_FACTOR: 1.0
  WARMUP_ITERS: 10
  LR_SCHEDULER_NAME: "WarmupPolyLR"
  WEIGHT_DECAY: 0.05
  WEIGHT_DECAY_NORM: 0.0
  WEIGHT_DECAY_EMBED: 0.0
  BACKBONE_MULTIPLIER: 1.0
  TEST_IMS_PER_BATCH: 1
  CHECKPOINT_PERIOD: 5000
  AMP:
    ENABLED: False
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "full_model"
    CLIP_VALUE: 0.01
    NORM_TYPE: 2.0
INPUT:
  MIN_SIZE_TRAIN: !!python/object/apply:eval ["[int(x * 0.1 * 640) for x in range(5, 21)]"]
  MIN_SIZE_TRAIN_SAMPLING: "choice"
  MIN_SIZE_TEST: 640
  MAX_SIZE_TRAIN: 2560
  MAX_SIZE_TEST: 2560
  CROP:
    ENABLED: True
    TYPE: "absolute"
    SIZE: (640, 640)
    SINGLE_CATEGORY_MAX_AREA: 1.0
  COLOR_AUG_SSD: True
  SIZE_DIVISIBILITY: 640  # used in dataset mapper
  FORMAT: "RGB"
TEST:
  EVAL_PERIOD: 5000
  AUG:
    ENABLED: False
    MIN_SIZES: [256, 384, 512, 640, 768, 896]
    MAX_SIZE: 3584
    FLIP: True
DATALOADER:
  FILTER_EMPTY_ANNOTATIONS: True
  NUM_WORKERS: 4
VERSION: 2
