#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from pathlib import Path
import hashlib


def get_file_hash(filepath):
    """计算文件的MD5哈希值，用于检测重复图片"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


def process_markdown_files(root_dir):
    """
    处理指定目录下所有的Markdown文件
    将C盘Typora图片复制到本地image文件夹并更新路径
    """

    # 转换为Path对象
    root_path = Path(root_dir)

    # 创建image文件夹
    image_dir = root_path / "image"
    image_dir.mkdir(exist_ok=True)

    # 统计信息
    processed_files = 0
    copied_images = 0
    updated_files = []
    failed_images = []

    # 用于记录已复制图片的哈希值，避免重复复制相同内容的图片
    image_hash_map = {}

    # 正则表达式匹配Markdown中的图片
    # 匹配格式: ![alt](path) 或 <img src="path">
    img_pattern1 = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    img_pattern2 = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']')

    # 遍历所有md文件
    for md_file in root_path.rglob("*.md"):
        print(f"处理文件: {md_file.relative_to(root_path)}")

        try:
            # 读取文件内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            file_modified = False

            # 处理 ![alt](path) 格式的图片
            for match in img_pattern1.finditer(original_content):
                alt_text = match.group(1)
                img_path = match.group(2)

                # 检查是否是C盘Typora路径
                if 'AppData\\Roaming\\Typora\\typora-user-images' in img_path or \
                        'AppData/Roaming/Typora/typora-user-images' in img_path:

                    # 处理路径中的反斜杠
                    img_path_normalized = img_path.replace('/', '\\')

                    # 获取图片文件名
                    img_name = os.path.basename(img_path_normalized)

                    # 源文件路径
                    source_path = Path(img_path_normalized)

                    # 目标文件路径
                    target_path = image_dir / img_name

                    # 复制图片
                    if source_path.exists():
                        # 计算源文件哈希
                        source_hash = get_file_hash(source_path)

                        # 检查是否已有相同内容的图片
                        if source_hash and source_hash in image_hash_map:
                            # 使用已存在的图片
                            existing_img = image_hash_map[source_hash]
                            target_path = image_dir / existing_img
                            print(f"  图片内容已存在，使用: {existing_img}")
                        else:
                            # 如果目标文件已存在但内容不同，重命名
                            if target_path.exists():
                                target_hash = get_file_hash(target_path)
                                if target_hash != source_hash:
                                    # 添加数字后缀
                                    base_name = target_path.stem
                                    extension = target_path.suffix
                                    counter = 1
                                    while target_path.exists():
                                        new_name = f"{base_name}_{counter}{extension}"
                                        target_path = image_dir / new_name
                                        counter += 1
                                    img_name = target_path.name

                            # 复制文件
                            shutil.copy2(source_path, target_path)
                            copied_images += 1
                            print(f"  已复制图片: {img_name}")

                            # 记录哈希值
                            if source_hash:
                                image_hash_map[source_hash] = img_name

                        # 计算相对路径
                        # 获取md文件相对于根目录的路径深度
                        md_relative = md_file.relative_to(root_path)
                        depth = len(md_relative.parts) - 1  # 减1是因为最后一个是文件名

                        # 构建相对路径
                        if depth == 0:
                            # md文件在根目录
                            new_path = f"./image/{img_name}"
                        else:
                            # md文件在子目录中，需要返回上级目录
                            back_path = "../" * depth
                            new_path = f"{back_path}image/{img_name}"

                        # 替换路径
                        old_str = match.group(0)
                        new_str = f"![{alt_text}]({new_path})"
                        content = content.replace(old_str, new_str)
                        file_modified = True

                    else:
                        print(f"  警告: 图片文件不存在: {source_path}")
                        failed_images.append(str(source_path))

            # 处理 <img src="path"> 格式的图片
            for match in img_pattern2.finditer(original_content):
                img_path = match.group(1)

                # 检查是否是C盘Typora路径
                if 'AppData\\Roaming\\Typora\\typora-user-images' in img_path or \
                        'AppData/Roaming/Typora/typora-user-images' in img_path:

                    # 处理路径中的反斜杠
                    img_path_normalized = img_path.replace('/', '\\')

                    # 获取图片文件名
                    img_name = os.path.basename(img_path_normalized)

                    # 源文件路径
                    source_path = Path(img_path_normalized)

                    # 目标文件路径
                    target_path = image_dir / img_name

                    # 复制图片（逻辑同上）
                    if source_path.exists():
                        source_hash = get_file_hash(source_path)

                        if source_hash and source_hash in image_hash_map:
                            existing_img = image_hash_map[source_hash]
                            target_path = image_dir / existing_img
                        else:
                            if target_path.exists():
                                target_hash = get_file_hash(target_path)
                                if target_hash != source_hash:
                                    base_name = target_path.stem
                                    extension = target_path.suffix
                                    counter = 1
                                    while target_path.exists():
                                        new_name = f"{base_name}_{counter}{extension}"
                                        target_path = image_dir / new_name
                                        counter += 1
                                    img_name = target_path.name

                            shutil.copy2(source_path, target_path)
                            copied_images += 1
                            print(f"  已复制图片: {img_name}")

                            if source_hash:
                                image_hash_map[source_hash] = img_name

                        # 计算相对路径
                        md_relative = md_file.relative_to(root_path)
                        depth = len(md_relative.parts) - 1

                        if depth == 0:
                            new_path = f"./image/{img_name}"
                        else:
                            back_path = "../" * depth
                            new_path = f"{back_path}image/{img_name}"

                        # 替换路径
                        content = content.replace(img_path, new_path)
                        file_modified = True

                    else:
                        print(f"  警告: 图片文件不存在: {source_path}")
                        failed_images.append(str(source_path))

            # 如果文件有修改，保存更改
            if file_modified:
                # 备份原文件
                backup_path = md_file.with_suffix('.md.bak')
                shutil.copy2(md_file, backup_path)

                # 写入新内容
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                updated_files.append(str(md_file.relative_to(root_path)))
                processed_files += 1
                print(f"  已更新文件并创建备份: {backup_path.name}")

        except Exception as e:
            print(f"  处理文件时出错: {e}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成！统计信息：")
    print(f"更新的文件数: {processed_files}")
    print(f"复制的图片数: {copied_images}")

    if updated_files:
        print("\n已更新的文件列表:")
        for file in updated_files:
            print(f"  - {file}")

    if failed_images:
        print("\n未找到的图片文件:")
        for img in failed_images:
            print(f"  - {img}")

    print("\n注意：原始文件已备份为 .md.bak 文件")
    print("如需恢复，请删除修改后的.md文件并将.bak文件重命名")


def main():
    """主函数"""
    print("Typora图片路径修复工具")
    print("=" * 50)

    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 确认操作
    confirm = input("\n是否开始处理？(y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    print("\n开始处理...\n")

    try:
        process_markdown_files(current_dir)
    except Exception as e:
        print(f"\n发生错误: {e}")
        return

    print("\n处理完成！")
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main()