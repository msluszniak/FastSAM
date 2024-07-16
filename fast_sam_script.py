from fastsam import FastSAM
import coremltools as ct
import torch


def main():
    fsam = FastSAM("weights/FastSAM-s.pt")
    fsam.export(format="torchscript")
    imgsz = fsam.overrides["imgsz"]

    ts = torch.load("weights/FastSAM-s.torchscript")

    model = ct.convert(
        ts,
        inputs=[
            ct.ImageType(
                name="image", scale=1 / 255, bias=(0, 0, 0), shape=(1, 3, imgsz, imgsz)
            )
        ],
        convert_to="mlprogram",
    )
    model.save("weights/FastSAM-s.mlpackage")


if __name__ == "__main__":
    main()
