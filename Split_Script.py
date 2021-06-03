import os
from ComputationalTools import *
import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--script_folder", type=str, default="Scripts", help="script_folder")
parser.add_argument("--script_name", type=str, default="", help="script_name")
parser.add_argument("--gpu_num", type=int, default=8, help="gpu_num")
parser.add_argument("--gpu_id_list", type=str, default="", help="gpu_id_list")
parser.add_argument("--keep_old_gpu_id", action='store_true', help="keep_old_gpu_id")
args = parser.parse_args()

script_folder = args.script_folder
if args.script_name == "":
    script_name = Get_latest_files("Scripts", delimiter="sh")[0]
else:
    script_name = args.script_name
gpu_num = args.gpu_num

if args.gpu_id_list == "":
    gpu_id_list = list(range(gpu_num))
else:
    gpu_id_list = args.gpu_id_list.split(",")

if args.keep_old_gpu_id:
    reassign_gpu_id = False
else:
    reassign_gpu_id = True


print("script_folder:", script_folder)
print("script_name:", script_name)
print("reassign_gpu_id:", reassign_gpu_id)
print("gpu_num:", gpu_num)
print("gpu_id_list:", gpu_id_list)
# exit()

# CAS-5544084-G3V0K1

gpu_order_list = list(range(gpu_num))

script_line_dict = {}
script_line_list = []
with open(os.path.join(script_folder, f"{script_name}.sh"), "r") as f:

    file_line_list = f.readlines()
    # print(file_line_list)

    for line in file_line_list:
        if line.split(" ")[0].split("=")[0] == "CUDA_VISIBLE_DEVICES":
            line = " ".join(line.split(" ")[1:])

        script_line_list.append(line)
        if not reassign_gpu_id:
            try:
                gpu_id = line.split(" ")[0].split("=")[1]
                print(gpu_id, end="")
            except:
                continue

            if gpu_id not in script_line_dict:
                script_line_dict[gpu_id] = [line]
            else:
                script_line_dict[gpu_id].append(line)

print("")
if reassign_gpu_id:
    for gpu_id, gpu_order in zip(gpu_id_list, gpu_order_list):
        start_pos, end_pos = Get_start_and_end_pos_for_worker(gpu_order, gpu_num, 0, len(script_line_list))
        print(f"gpu_id {gpu_id}, order {gpu_order}, [{start_pos}, {end_pos}]")
        if end_pos - start_pos > 0:
            filename = f"{script_name}_g{gpu_id}.sh"
            print(f"Created {filename}")
            with open(os.path.join(script_folder, filename), "w") as g:
                for line in script_line_list[start_pos:end_pos]:
                    g.write(f"CUDA_VISIBLE_DEVICES={gpu_id} {line}")


else:
    for gpu_id in script_line_dict:
        filename = f"{script_name}_g{gpu_id}.sh"
        print(f"Created {filename}")
        with open(os.path.join(script_folder, filename), "w") as g:
            for line in script_line_dict[gpu_id]:
                g.write(line)