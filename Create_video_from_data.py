from deepviz import *
from deepviz_1D import DeepVisuals_1D
from deepviz_pre import DeepVisuals_Pre
from deepviz_summary import *
from deepviz_summary_II import DeepVisuals_Summary_II
from deepviz_summary_weighted_BP_KDE_and_grad_norm import DeepVisuals_Summary_weighted_BP_KDE_and_grad_norm
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import os
import glob
import argparse
import datetime

# from utils.logger import SendMail

try:
    import winsound  # for sound
    from sound import  *
except:
    print("Not using Windows")

from multiprocessing import Pool
""" Use multiprocessing """
parser = argparse.ArgumentParser()
parser.add_argument("--latest_file_num", "--lfn", type=int, default=1, help="number of files for video")
parser.add_argument("--skip_num", "--sn", type=int, default=0, help="number of files skipped")
parser.add_argument("--total_parts", "--tp", type=int, default=6, help="number of workers")
parser.add_argument("--skip_frame", "--sf", type=int, default=1, help="number of iterations used for one frame")
parser.add_argument("--try2", action='store_true', help="whether to try")
parser.add_argument("--task", type=str, default="", help="task filename")
parser.add_argument("--task_folder", type=str, default="Data", help="task folder")
parser.add_argument("--type", type=str, default="2D", help="deepviz type")
parser.add_argument("--fps", type=int, default=10, help="fps")
parser.add_argument("--data", type=str, default="", help="dataset name")
parser.add_argument("--cumul", action='store_true', help="whether to cumulate updates")

args = parser.parse_args()
print(args)
# args.latest_file_num = 1

total_parts = args.total_parts
def Write_video_part_to_file(task_filename, my_part, num_parts, skip_frame=1, dataset_name=""):
    iter_start = 0
    iter_end = np.inf
    # iter_start = 30000
    # iter_end = 0

    if args.type in ["2D", "2d"]:
        deepVisuals = DeepVisuals_2D(data_folder=args.task_folder)
        deepVisuals.Generate_video_from_file(task_filename, my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        # deepVisuals.name += f"_is{iter_start}_ie{iter_end}"
        deepVisuals.Save_plot(fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    if args.type in ["1D", "1d"]:
        deepVisuals = DeepVisuals_1D()
        deepVisuals.Generate_video_from_file(task_filename, my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        # deepVisuals.name += f"_is{iter_start}_ie{iter_end}"
        deepVisuals.Save_plot(fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    elif args.type == "pre":
        deepVisuals = DeepVisuals_Pre()
        deepVisuals.Generate_video_from_file(task_filename, my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        deepVisuals.Save_plot(fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    elif args.type == "sum":
        if dataset_name == "":
            print("Please provide a dataset name.")
            raise ValueError

        deepVisuals = DeepVisuals_Summary(task_filename, dataset_name, args.cumul)
        deepVisuals.Generate_summary_video_from_file(my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        deepVisuals.Save_plot(figures_folder=summary_dir, fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    elif args.type == "sum2":
        if dataset_name == "":
            print("Please provide a dataset name.")
            raise ValueError

        deepVisuals = DeepVisuals_Summary_II(task_filename, dataset_name, args.cumul)
        deepVisuals.Generate_summary_video_from_file(my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        deepVisuals.Save_plot(figures_folder=summary_dir, fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    elif args.type == "sum3":
        if dataset_name == "":
            print("Please provide a dataset name.")
            raise ValueError

        deepVisuals = DeepVisuals_Summary_weighted_BP_KDE_and_grad_norm(task_filename, dataset_name)
        deepVisuals.Generate_summary_video_from_file(my_part, num_parts, iter_start=iter_start, iter_end=iter_end, skip_frame=skip_frame)
        deepVisuals.Save_plot(figures_folder=os.path.join(summary_dir, task_filename), fps=args.fps)
        if deepVisuals.total_frame == 0:
            return ""
        else:
            return deepVisuals.attr["name"]

    else:
        print("Unknown args.type")
        raise NotImplementedError

if __name__ == "__main__":
    st = time.time()

    video_folder = figures_folder
    cumul_text = ""
    if args.type in ["sum", "sum2", "sum3"]:
        video_folder = summary_dir
        if args.cumul:
            cumul_text = "_cumul"
    if args.task == "":
        if args.type in ["sum", "sum2", "sum3"]:
            print("Please provide a task name.")
            raise ValueError

        task_list = Get_latest_files(args.task_folder, args.latest_file_num, args.skip_num, "pickle")
    else:
        task_list = args.task.split(",")

    if args.type == "sum3":
        task_list = [args.task]
    print(task_list)


    for task_filename in task_list:
        save_plot_start_time = time.time()

        if not args.try2:
            clip_list = None
            with Pool() as p:
                clip_list = p.starmap(Write_video_part_to_file, [(task_filename, i, total_parts, args.skip_frame, args.data) for i in range(total_parts)])
                print(clip_list)

            print(f"video saving time: {time.time() - save_plot_start_time}")
            if len(clip_list) > 1:
                video_merge_start_time = time.time()

                video_clip_list = []
                for clip in clip_list:
                    if clip != "" and clip is not None:
                        clip_filename = os.path.join(video_folder, clip + ".mp4")

                        if args.try2:
                            try:
                                video_clip = VideoFileClip(clip_filename)
                                print(f"{clip_filename} duration {video_clip.duration:.3f}")
                                if hasattr(video_clip, "duration"):
                                    if video_clip.duration > 0:
                                        video_clip_list.append(video_clip)
                            except:
                                print(f"Clip error {clip_filename}")
                        else:
                            video_clip = VideoFileClip(clip_filename)
                            print(f"{clip_filename} duration {video_clip.duration:.3f}")
                            if hasattr(video_clip, "duration"):
                                if video_clip.duration > 0:
                                    video_clip_list.append(video_clip)

                if len(video_clip_list) > 0:
                    final_clip = concatenate_videoclips(video_clip_list)
                    final_title = task_filename + f"{cumul_text}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
                    print("title length =", len(final_title))
                    title_len_thresh = 220
                    if len(final_title) > title_len_thresh:
                        final_title = final_title[:title_len_thresh]
                    final_clip.write_videofile(
                        os.path.join(video_folder, f"{final_title}.mp4"))
                else:
                    print("No clip available.")

                [os.remove(os.path.join(video_folder, clip + ".mp4")) for clip in clip_list if clip != "" and clip is not None]
        else:
            try:
                clip_list = None
                with Pool() as p:
                    clip_list = p.starmap(Write_video_part_to_file, [(task_filename, i, total_parts, args.skip_frame, args.data) for i in range(total_parts)])
                    print(clip_list)

                print(f"video saving time: {time.time() - save_plot_start_time}")
                if len(clip_list) > 1:
                    video_merge_start_time = time.time()

                    video_clip_list = []
                    for clip in clip_list:
                        if clip != "" and clip is not None:
                            clip_filename = os.path.join(video_folder, clip + ".mp4")

                            if args.try2:
                                try:
                                    video_clip = VideoFileClip(clip_filename)
                                    print(f"{clip_filename} duration {video_clip.duration:.3f}")
                                    if hasattr(video_clip, "duration"):
                                        if video_clip.duration > 0:
                                            video_clip_list.append(video_clip)
                                except:
                                    print(f"Clip error {clip_filename}")
                            else:
                                video_clip = VideoFileClip(clip_filename)
                                print(f"{clip_filename} duration {video_clip.duration:.3f}")
                                if hasattr(video_clip, "duration"):
                                    if video_clip.duration > 0:
                                        video_clip_list.append(video_clip)

                    if len(video_clip_list) > 0:
                        final_clip = concatenate_videoclips(video_clip_list)
                        final_title = task_filename + f"{cumul_text}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
                        print("title length =", len(final_title))
                        title_len_thresh = 220
                        if len(final_title) > title_len_thresh:
                            final_title = final_title[:title_len_thresh]
                        final_clip.write_videofile(os.path.join(video_folder, f"{final_title}.mp4"))
                    else:
                        print("No clip available.")

                    [os.remove(os.path.join(video_folder, clip + ".mp4")) for clip in clip_list if clip != "" and clip is not None]
                    print(f"video merge time: {time.time() - video_merge_start_time}")
                pass
            except:
                print(f"[Task failed] {task_filename}")
        print(f"total time: {time.time() - save_plot_start_time:.5f}")

        # SendMail("juyilong@gmail.com", "juyilong@gmail.com", task_filename + f"{time.strftime('%Y%m%d_%H%M%S')}.mp4", task_filename + f"{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    print(f"Run time: {time.time() - st:.5f}")

    try:
        winsound.Beep(400, 250)  # frequency, duration
        time.sleep(0.25)  # in seconds (0.25 is 250ms)
        winsound.Beep(600, 1000)
        time.sleep(1)
        # Cir()
        # End()
        pass
    except:
        print("Not using windows.")

