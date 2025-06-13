import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BlazeBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # 根据stride决定padding
        padding = 0 if stride == 2 else (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias_attr=True,
            ),
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True,
            ),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        if self.stride == 2:
            # 使用固定形状的padding
            batch_size = 1  # 固定批量大小为1
            channels = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]

            # 手动实现padding
            padding_h = paddle.zeros([batch_size, channels, height, 2], dtype=x.dtype)
            padding_v = paddle.zeros(
                [batch_size, channels, 2, width + 2], dtype=x.dtype
            )
            h = paddle.concat([x, padding_h], axis=3)  # 水平padding
            h = paddle.concat([h, padding_v], axis=2)  # 垂直padding

            # max pooling
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        else:
            h = x

        if self.channel_pad > 0:
            # 使用固定形状的通道padding
            batch_size = 1  # 固定批量大小为1
            height = x.shape[2]
            width = x.shape[3]

            x = paddle.concat(
                [
                    x,
                    paddle.zeros(
                        [batch_size, self.channel_pad, height, width], dtype=x.dtype
                    ),
                ],
                axis=1,
            )

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Layer):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=2,
                padding=0,
                groups=channels,
                bias_attr=True,
            ),
            nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True,
            ),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        # 使用固定形状的padding
        batch_size = 1  # 固定批量大小为1
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        # 手动实现padding
        padding_h = paddle.zeros([batch_size, channels, height, 2], dtype=x.dtype)
        padding_v = paddle.zeros([batch_size, channels, 2, width + 2], dtype=x.dtype)
        h = paddle.concat([x, padding_h], axis=3)  # 水平padding
        h = paddle.concat([h, padding_v], axis=2)  # 垂直padding

        return self.act(self.convs(h))


class BlazeFaceFixed(nn.Layer):
    """固定形状的BlazeFace模型，仅适用于batch_size=1的情况"""

    def __init__(self):
        super(BlazeFaceFixed, self).__init__()

        # 参数设置
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0

        # 设置缩放参数
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0

        # 定义网络层
        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2D(
                in_channels=3,
                out_channels=24,
                kernel_size=5,
                stride=2,
                padding=2,
                bias_attr=True,
            ),
            nn.ReLU(),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24, stride=2),  # 输出shape: [1, 24, 64, 64]
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),  # 输出shape: [1, 48, 32, 32]
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 96, stride=2),  # 输出shape: [1, 96, 16, 16]
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),  # 输出shape: [1, 96, 16, 16]
        )

        self.final = FinalBlazeBlock(96)  # 输出shape: [1, 96, 8, 8]

        self.classifier_8 = nn.Conv2D(
            96, 2, 1, bias_attr=True
        )  # 输出shape: [1, 2, 16, 16]
        self.classifier_16 = nn.Conv2D(
            96, 6, 1, bias_attr=True
        )  # 输出shape: [1, 6, 8, 8]

        self.regressor_8 = nn.Conv2D(
            96, 32, 1, bias_attr=True
        )  # 输出shape: [1, 32, 16, 16]
        self.regressor_16 = nn.Conv2D(
            96, 96, 1, bias_attr=True
        )  # 输出shape: [1, 96, 8, 8]

    def forward(self, x):
        # 输入shape: [1, 3, 256, 256]
        x = self.backbone(x)  # 输出shape: [1, 96, 16, 16]
        h = self.final(x)  # 输出shape: [1, 96, 8, 8]

        # 使用固定形状处理分类器输出
        c1 = self.classifier_8(x)  # shape: [1, 2, 16, 16]
        c1 = paddle.transpose(c1, [0, 2, 3, 1])  # shape: [1, 16, 16, 2]
        c1 = paddle.reshape(c1, [1, 512, 1])  # shape: [1, 512, 1]

        c2 = self.classifier_16(h)  # shape: [1, 6, 8, 8]
        c2 = paddle.transpose(c2, [0, 2, 3, 1])  # shape: [1, 8, 8, 6]
        c2 = paddle.reshape(c2, [1, 384, 1])  # shape: [1, 384, 1]

        c = paddle.concat([c1, c2], axis=1)  # shape: [1, 896, 1]

        # 使用固定形状处理回归器输出
        r1 = self.regressor_8(x)  # shape: [1, 32, 16, 16]
        r1 = paddle.transpose(r1, [0, 2, 3, 1])  # shape: [1, 16, 16, 32]
        r1 = paddle.reshape(r1, [1, 512, 16])  # shape: [1, 512, 16]

        r2 = self.regressor_16(h)  # shape: [1, 96, 8, 8]
        r2 = paddle.transpose(r2, [0, 2, 3, 1])  # shape: [1, 8, 8, 96]
        r2 = paddle.reshape(r2, [1, 384, 16])  # shape: [1, 384, 16]

        r = paddle.concat([r1, r2], axis=1)  # shape: [1, 896, 16]

        return [r, c]

    def set_anchors(self, anchors):
        """设置锚框"""
        if isinstance(anchors, np.ndarray):
            self.anchors = paddle.to_tensor(anchors, dtype="float32")
        else:
            self.anchors = anchors

        assert len(self.anchors.shape) == 2
        assert self.anchors.shape[0] == self.num_anchors
        assert self.anchors.shape[1] == 4

    def load_weights_from_pytorch(self, weights_path):
        """从PyTorch权重文件加载权重到Paddle模型"""
        import torch

        pytorch_state_dict = torch.load(weights_path, map_location="cpu")
        paddle_state_dict = {}

        # 打印PyTorch权重的部分键值，用于调试
        print("PyTorch权重键示例:")
        for i, key in enumerate(list(pytorch_state_dict.keys())[:5]):
            print(f"  {key}: {pytorch_state_dict[key].shape}")

        # 打印Paddle模型的部分键值
        paddle_model_state = self.state_dict()
        print("Paddle模型键示例:")
        for i, key in enumerate(list(paddle_model_state.keys())[:5]):
            print(f"  {key}: {paddle_model_state[key].shape}")

        # 对每个PyTorch权重进行处理
        for key, value in pytorch_state_dict.items():
            value_np = value.numpy()

            # 处理具有4个维度的权重 (卷积核)
            if key in paddle_model_state and len(value_np.shape) == 4:
                paddle_state_dict[key] = paddle.to_tensor(value_np)
                print(
                    f"转换权重 {key}: {value_np.shape} -> {paddle_state_dict[key].shape}"
                )

            # 处理具有2个维度或1个维度的权重 (全连接层权重或偏置)
            elif key in paddle_model_state and (
                len(value_np.shape) == 2 or len(value_np.shape) == 1
            ):
                # 对于fc层权重，通常需要转置
                if len(value_np.shape) == 2:
                    value_np = value_np.transpose()

                paddle_state_dict[key] = paddle.to_tensor(value_np)

            # 其他情况
            elif key in paddle_model_state:
                paddle_state_dict[key] = paddle.to_tensor(value_np)
            else:
                print(f"跳过未知权重: {key}")

        # 更新模型权重
        self.set_state_dict(paddle_state_dict)
        print("权重加载完成")
        return True


# 导出模型类
class BlazeFaceExport(nn.Layer):
    def __init__(self, blazeface_model):
        super(BlazeFaceExport, self).__init__()
        self.model = blazeface_model

    def forward(self, x):
        # 直接使用模型，不做任何预处理或后处理
        return self.model(x)
