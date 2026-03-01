import os
import argparse
import xml.etree.ElementTree as ET
from mmdet.datasets.voc import class_voc, class_dior, class_dota


all_class = {"DIOR": class_dior, "DOTA": class_dota, "VOC": class_voc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VOC-style dataset into incremental tasks")
    parser.add_argument("-p", "--patter", type=str, default="15+1")
    parser.add_argument("-d", "--dataset", type=str, default="DIOR", choices=["DIOR", "DOTA", "VOC"])
    args = parser.parse_args()

    patter = args.patter
    dataset = args.dataset
    data_root = {
        "DIOR": "/data/my_code/dataset/DIOR",
        "DOTA": "/data/my_code/dataset/DOTA_xml",
        "VOC": "/data/my_code/dataset/VOC",
    }[dataset]
    all_class = all_class[dataset]

    input_dir_trainval = f"{data_root}/ImageSets/Main/trainval.txt"
    input_dir_test = f"{data_root}/ImageSets/Main/test.txt"
    pattern_root = os.path.join(data_root, patter)
    ann_dir = os.path.join(data_root, "Annotations")

    base = int(patter.split("+")[0])
    class_increment = int(patter.split("+")[1])

    def read_filenames(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def parse_data(xml_path):
        if not os.path.exists(xml_path):
            return set()
        tree = ET.parse(xml_path)
        root = tree.getroot()
        classes = set()
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is not None and name_node.text:
                classes.add(name_node.text.strip())
        return classes

    def ensure_parent(path):
        os.makedirs(path, exist_ok=True)

    def write_filtered_xml(src_xml, dst_xml, keep_class_set):
        tree = ET.parse(src_xml)
        root = tree.getroot()
        objects_to_remove = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            name = name_node.text.strip() if name_node is not None and name_node.text else ""
            if name not in keep_class_set:
                objects_to_remove.append(obj)

        for obj in objects_to_remove:
            root.remove(obj)

        if root.findall("object"):
            tree.write(dst_xml)
            return True
        return False

    def dump_split(task_id, split_name, keep_ids, keep_class_set):
        ann_out_dir = os.path.join(pattern_root, f"task{task_id}_{split_name}")
        ensure_parent(ann_out_dir)

        txt_path = os.path.join(pattern_root, f"task{task_id}_{split_name}.txt")
        written_ids = []

        for filename in keep_ids:
            src_xml = os.path.join(ann_dir, f"{filename}.xml")
            dst_xml = os.path.join(ann_out_dir, os.path.basename(src_xml))
            if write_filtered_xml(src_xml, dst_xml, keep_class_set):
                written_ids.append(filename)

        with open(txt_path, "w", encoding="utf-8") as f:
            for filename in written_ids:
                f.write(filename + "\n")

        return ann_out_dir, txt_path, written_ids

    def collect_keep_list(filenames, task_class_set):
        keep = []
        for filename in filenames:
            xml_path = os.path.join(ann_dir, f"{filename}.xml")
            classes = parse_data(xml_path)
            if classes & task_class_set:
                keep.append(filename)
        return keep

    def process_split(task_id, split_name, filenames, task_class_set):
        keep = collect_keep_list(filenames, task_class_set)
        out_dir, out_txt, written_ids = dump_split(task_id, split_name, keep, task_class_set)
        return written_ids, out_dir, out_txt

    trainval_filenames = read_filenames(input_dir_trainval)
    test_filenames = read_filenames(input_dir_test)

    ensure_parent(pattern_root)

    class_ranges = []
    start = base
    task_idx = 1
    while start < len(all_class):
        end = min(start + class_increment, len(all_class))
        class_ranges.append((task_idx, all_class[start:end], all_class[:end]))
        start = end
        task_idx += 1

    print(f"dataset={dataset}, pattern={patter}, total_classes={len(all_class)}, total_tasks={len(class_ranges)}")

    for task_id, task_classes, cumulative_classes in class_ranges:
        task_class_set = set(task_classes)
        cumulative_class_set = set(cumulative_classes)

        selected_trainval, trainval_dir, trainval_txt = process_split(
            task_id, "trainval", trainval_filenames, task_class_set
        )
        selected_test, test_dir, test_txt = process_split(
            task_id, "test", test_filenames, cumulative_class_set
        )

        print(
            f"trainval={len(selected_trainval)}, test={len(selected_test)}"
        )
        print(f"  -> {trainval_dir}")
        print(f"  -> {trainval_txt}")
        print(f"  -> {test_dir}")
        print(f"  -> {test_txt}")
