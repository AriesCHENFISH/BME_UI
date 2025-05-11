import os
import torch
import argparse
from data import load_samples, get_dataloaders
from model import get_model, get_loss_functions
from train import train_model_kfold
from predict import test_model, load_model_for_inference, predict_single_image
from utils import set_seed, split_train_test


def parse_args():
    parser = argparse.ArgumentParser(description='超声图像分类和分割模型')

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'predict'],
                        help='运行模式: train (训练), test (测试), predict (预测)')

    parser.add_argument('--data_dir', type=str, default='/autodl-fs/data/DeepLearning_SRTP/data',
                        help='数据根目录')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')

    parser.add_argument('--model_path', type=str,
                        help='模型路径，用于测试或预测模式')

    parser.add_argument('--image_path', type=str,
                        help='图像路径，用于预测单张图像')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率')

    parser.add_argument('--epochs', type=int, default=80,
                        help='最大训练轮数')

    parser.add_argument('--patience', type=int, default=15,
                        help='早停patience')

    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    parser.add_argument('--n_splits', type=int, default=5,
                        help='交叉验证折数')

    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集比例')

    parser.add_argument('--use_cuda', action='store_true',
                        help='使用CUDA')

    return parser.parse_args()


def train_mode(args):
    # 设置随机种子
    set_seed(args.seed)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载样本
    all_samples = load_samples(args.data_dir)
    train_samples, test_samples = split_train_test(all_samples, args.test_ratio)
    print(f"训练集数量: {len(train_samples)}, 测试集数量: {len(test_samples)}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"使用设备: {device}")

    # 创建测试数据加载器
    _, _, test_loader = get_dataloaders([], [], test_samples, args.batch_size)

    # 获取损失函数
    criterion_cls, criterion_seg = get_loss_functions()

    # 训练模型
    train_model_kfold(
        lambda dev: get_model(dev),
        train_samples,
        test_loader,
        criterion_cls,
        criterion_seg,
        device,
        args.output_dir,
        args.n_splits,
        args.batch_size,
        args.lr
    )


def test_mode(args):
    if not args.model_path:
        raise ValueError("测试模式需要指定--model_path")

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"使用设备: {device}")

    # 加载样本
    all_samples = load_samples(args.data_dir)
    _, test_samples = split_train_test(all_samples, args.test_ratio)
    print(f"测试集数量: {len(test_samples)}")

    # 创建测试数据加载器
    _, _, test_loader = get_dataloaders([], [], test_samples, args.batch_size)

    # 加载模型
    model = load_model_for_inference(args.model_path, device)

    # 测试模型
    results_dir = os.path.join(args.output_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou = test_model(model, test_loader, device, results_dir)

    print(f"测试结果:")
    print(f"分类 - 准确率: {acc_cls:.4f}, 召回率: {recall_cls:.4f}, F1分数: {f1_cls:.4f}")
    print(f"分割 - 像素准确率: {pixel_acc:.4f}, IoU: {iou:.4f}")

    # 保存结果
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"分类 - 准确率: {acc_cls:.4f}, 召回率: {recall_cls:.4f}, F1分数: {f1_cls:.4f}\n")
        f.write(f"分割 - 像素准确率: {pixel_acc:.4f}, IoU: {iou:.4f}\n")
        f.write(f"混淆矩阵:\n{cm_cls}\n")


def predict_mode(args):
    if not args.model_path:
        raise ValueError("预测模式需要指定--model_path")
    if not args.image_path:
        raise ValueError("预测模式需要指定--image_path")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model_for_inference(args.model_path, device)

    # 预测
    results_dir = os.path.join(args.output_dir, 'predict_results')
    os.makedirs(results_dir, exist_ok=True)

    cls_pred, seg_pred = predict_single_image(model, args.image_path, device, results_dir)

    print(f"预测结果:")
    print(f"分类: {'恶性' if cls_pred == 1 else '良性'}")
    print(f"分割结果已保存至: {results_dir}")


def main():
    args = parse_args()

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        raise ValueError(f"无效的模式: {args.mode}")


if __name__ == "__main__":
    main()