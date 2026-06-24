import cv2
import torch
import utils
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def compose2(f1, f2):
    return lambda x: f2(f1(x))


model_params = (
    {"name": "dpt_large-midas", "path": "weights/dpt_large-midas-2f21e586.pt", "backbone": "vitl16_384"},
    {"name": "dpt_hybrid-midas", "path": "weights/dpt_hybrid-midas-501f0c75.pt", "backbone": "vitb_rn50_384"}
)

for model_param in model_params:
    model_path = model_param["path"]
    device = torch.device("cpu")
    model = DPTDepthModel(
        path=model_path,
        backbone=model_param["backbone"],
        non_negative=True,
    )
    net_w, net_h = 384, 384
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    resize_image = Resize(
        net_w,
        net_h,
        resize_target=None,
        keep_aspect_ratio=False,
        ensure_multiple_of=32,
        resize_method="upper_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )

    transform = Compose(
        [
            resize_image,
            normalization,
            PrepareForNet()
        ]
    )
    model.eval()

    img = utils.read_image("input/dog.jpg")
    img_input = transform({"image": img})["image"]
    shaped = img_input.reshape(1, 3, net_h, net_w)
    torch.onnx.export(model, torch.rand(1, 3, 384, 384, dtype=torch.float), "weights/" + model_param["name"] + ".onnx",
                      export_params=True)
