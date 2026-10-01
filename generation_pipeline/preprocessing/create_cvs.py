import pandas as pd
import json
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import os
import multiprocessing as mp
import warnings
import argparse 


warnings.filterwarnings("ignore")

def round_v2(el) :
    """Round value with fallback for invalid inputs."""
    try :
        return round(el)
    except :
        return -100

def process_json(input_list,overwrite=False):
    """Convert OCR JSON to CSV with Sauvola-based color estimation."""
    black,img_path=input_list
    img_path_full = os.path.abspath(img_path)
    
    if black :
        json_path=img_path.replace("_black.png",".json").replace("_black.jpeg",".json").replace("_black.jpg",".json").replace("images_black","merged_jsons")
    else :
        json_path=img_path.replace(".png",".json").replace(".jpeg",".json").replace(".jpg",".json").replace("images","merged_jsons")
    
    if black :
        csv_path=json_path.replace(".json","_black.csv")       
    else :
        csv_path=json_path.replace(".json",".csv")
    
    if os.path.exists(csv_path) and not overwrite:
        
        print(f"Path {csv_path} exists")
    else :
            
        try :

            with open(json_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame()

            img=cv2.imread(img_path_full)

        except Exception as e:
                print(e)
                return None

        for word in data:
            try :

                text = word["Word"]
                coord= word["BoundingBox"]

                W_d, W_g, H_b, H_h= coord["W_d"], coord["W_g"], coord["H_b"], coord["H_h"]

                coord=[W_d, W_g, H_b, H_h]
                coord_test = [W_d, W_g, H_b, H_h]
                if coord_test[2] < coord_test[3] and coord_test[0] < coord_test[1] :

                    try:
                        #Apply Sauvola to approximate color, necessary only in phase one and insertion
                        patch_test=img[coord_test[2]:coord_test[3], coord_test[0]:coord_test[1]]
                        gray = cv2.cvtColor (patch_test, cv2.COLOR_BGR2GRAY)

                        window_size = 21 

                        k = 0.5 
                        R = 100 
                        thresh_sauvola = threshold_sauvola(gray, window_size=window_size, k=k, r=R)

                        bg_color = (np.expand_dims((gray > thresh_sauvola),2)*patch_test).sum((0,1))/(gray > thresh_sauvola).sum((0,1))

                        R = 150 
                        thresh_sauvola = threshold_sauvola(gray, window_size=window_size, k=k, r=R)

                        fg_color= (np.expand_dims((gray < thresh_sauvola),2)*patch_test).sum((0,1))/(gray < thresh_sauvola).sum((0,1))

                        fg_color=[round_v2(el) for el in fg_color]
                        bg_color=[round_v2(el) for el in bg_color]     
                    except:
                        print("Sauvola failed, setting color to -100")
                        fg_color=[-100 for _ in range(3)]
                        bg_color=[-100 for _ in range(3)]

                    Width=W_g-W_d
                    Height=H_h-H_b
                    word_data = {
                        "text": text,
                        "filename": img_path_full,
                        "W_d": W_d,
                        "W_g": W_g,
                        "H_h": H_h,
                        "H_b": H_b,
                        "Width": Width,
                        "Height": Height,
                        "ratio" :round(Height/Width,2),
                        "fg_color" : [el for el in fg_color],
                        "bg_color" : [el for el in bg_color],
                        "len_text" : len(text)

                    }
                    
                    df = pd.concat([df, pd.DataFrame([word_data])], ignore_index=True)

            except Exception as e:
                    print(e)
                    return None

        if len(df)>0 :
            df.to_csv(csv_path)
            print(csv_path)


        return df

if __name__ == '__main__' :

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_folder", required=True)
    args = p.parse_args()
    dataset_folder=args.dataset_folder
    cpu_to_use=int(mp.cpu_count())

    for black in [False,True] :

        directory_path = os.path.join(dataset_folder,"images")
        if black :
            directory_path=directory_path.replace("images","images_black")

        image_paths=[]
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg") :
                    image_paths.append([black,filepath])
                if len(image_paths)>=cpu_to_use*10 :
                    pool = mp.Pool(cpu_to_use)
                    pool.map(process_json, image_paths)
                    pool.close()
                    pool.join()
                    image_paths=[]
        pool = mp.Pool(cpu_to_use)
        pool.map(process_json, image_paths)
        pool.close()
        pool.join()
