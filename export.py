import os
import numpy as np
import paddle
from blazeface_paddle import BlazeFaceFixed, BlazeFaceExport


def export_blazeface_model(save_dir="exported_model_fixed", model_prefix="inference"):
    """
    导出固定形状的BlazeFace模型为Paddle推理格式(pdmodel和pdparams)

    Args:
        save_dir: 保存导出模型的目录
        model_prefix: 模型文件前缀名
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化模型
    model = BlazeFaceFixed()

    # 从PyTorch权重加载
    print("从PyTorch权重加载...")
    model.load_weights_from_pytorch("blazefaceback.pth")

    # 加载锚框
    print("加载锚框...")
    anchors = np.load("anchorsback.npy")
    model.set_anchors(anchors)

    # 设置为评估模式
    model.eval()

    # 创建导出模型
    export_model = BlazeFaceExport(model)
    export_model.eval()

    # 设置模型输入，使用固定形状[1, 3, 256, 256]
    x_spec = paddle.static.InputSpec(shape=[1, 3, 256, 256], dtype="float32", name="x")

    # 保存模型
    save_path = os.path.join(save_dir, model_prefix)
    paddle.jit.save(export_model, save_path, input_spec=[x_spec])

    print(f"模型已导出到: {save_dir}")
    print(f"文件: {model_prefix}.pdmodel - 模型结构")
    print(f"文件: {model_prefix}.pdiparams - 模型参数")

    # 提供使用示例
    print("\n使用导出模型的示例代码:")
    print(
        """
import paddle
import cv2
import numpy as np

# 加载模型
model_path = "%s/%s"
model = paddle.jit.load(model_path)

# 准备输入数据 (注意：只支持batch_size=1)
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
x = paddle.to_tensor(img, dtype='float32')
x = paddle.transpose(x, [2, 0, 1])  # HWC -> CHW
x = paddle.unsqueeze(x, axis=0)     # 添加批次维度
x = x / 127.5 - 1.0                 # 预处理到[-1,1]范围

# 执行推理
raw_boxes, raw_scores = model(x)

# 然后可以应用后处理...
"""
        % (save_dir, model_prefix)
    )


if __name__ == "__main__":
    export_blazeface_model()
