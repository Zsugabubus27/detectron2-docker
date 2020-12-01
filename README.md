# Deep Sort with PyTorch 
### Detectron2 version of [DeepSORT PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

# Start Detectron Docker container
docker build -t detectron:latest .

docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume=$HOME/.torch/fvcore_cache:/tmp:rw --name=detectron-docker detectron:latest

docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume=$HOME/.torch/fvcore_cache:/tmp:rw detectron:latest python3 /home/appuser/detection/detection.py

docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume=$HOME/.torch/fvcore_cache:/tmp:rw -p 5555:5555 detectron:latest

python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input /tmp/sample_5k.bmp --output /tmp/outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
### Demo
1. Clone this repository: `git clone --recurse-submodules https://github.com/Zsugabubus27/detectron2-deepsort-pytorch.git`
2. Install torch!
3. Install detectron2: `cd detectron2-deepsort-pytorch` and `pip install -e detectron2/`
4. Install `deepsort` requirements: `pip install -r requirements.txt`
5. Run the demo: `python demo_detectron2_deepsort.py path/to/example_video.avi`

### Notes:
1. Try out different detectron2 models: Change the configs in `___init__` of `detectron2_detection.py`
2. Regarding any issues of detectron2, please refer to  [Detectron2](https://github.com/facebookresearch/detectron2) repository.

### References and Credits:
1. [Pytorch implementation of deepsort with Yolo3](https://github.com/ZQPei/deep_sort_pytorch)
2. [Facebook Detectron2](https://github.com/facebookresearch/detectron2)
3. [Deepsort](https://github.com/nwojke/deep_sort)
4. [SORT](https://github.com/abewley/sort)
