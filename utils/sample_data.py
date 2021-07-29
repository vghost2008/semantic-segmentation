import os
import glob
import shutil

def sample_dir(dir_path,save_dir,nr=10):
    if save_dir[-1] == "/":
        save_dir = save_dir[:-1]
    #save_dir = os.path.join(save_dir,os.path.basename(dir_path))
    prefix = os.path.basename(dir_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = glob.glob(os.path.join(dir_path,"*.jpg"))
    data_nr = len(files)
    delta = int(max(data_nr/nr,1))
    indexs = range(0,data_nr,delta)
    for i in indexs:
        src_path = files[i]
        save_path = os.path.join(save_dir,prefix+os.path.basename(src_path))
        shutil.copyfile(src_path,save_path)
    

def sample_dirs(dir_path,save_dir,nr=10):
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path,f) 
        if os.path.isdir(path):
            sample_dir(path,save_dir,nr)

if __name__ == "__main__":
    sample_dirs("/home/wj/ai/mldata/txc_park_videos/videos_rgb","/home/wj/ai/mldata/txc_park_videos/videos_rgb_10")