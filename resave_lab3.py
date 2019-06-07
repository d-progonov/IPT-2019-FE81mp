import os
import shutil
import re

def create_list_files(target_format_in, path_in):

    file_list = list()
    for root, _, files in os.walk(path_in):
        for curr_file in files:
            if target_format_in in curr_file:
                file_list.append(root + curr_file)
    return file_list

if __name__ == "__main__":
    file_list_tag = create_list_files(target_format_in='.txt',
                                   path_in='G:\\tags\\')
    file_list = create_list_files(target_format_in='.jpg',
                                      path_in='G:\\test\\')
    files_teg = os.listdir("G:\\tags\\")
    files = os.listdir("G:\\test\\")
    for key in range(0, len(files)):
        for j in range(0, len(files_teg)):
            if re.split('(\D+)', files_teg[j])[2] == re.split('(\D+)', files[key])[2]:
                shutil.move("G:\\tags\\" + files_teg[j], "G:\\tag\\tags" + re.split('(\D+)', files_teg[j])[2] + ".txt")
                break