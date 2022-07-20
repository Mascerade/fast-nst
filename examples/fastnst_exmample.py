from torch.optim import Adam
from nst.configs.fastnst_config import FastNSTConfig

config = FastNSTConfig(
    name="umbrella-girl",
    content_imgs_path="data/ms-coco",
    val_split=2000,
    epochs=2,
    batch_size=32,
    img_dim=(256, 256),
    style_img_path="example_data/umbrella-girl.jpg",
    content_layers={15: 1},
    style_layers={3: 0.3, 8: 0.7, 15: 0.7, 22: 0.3},
    lr=0.001,
    optimizer=Adam,
)

config.train()
