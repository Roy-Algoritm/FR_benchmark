"""

"""
import os
import shutil


def processor(data_path):
    print('begin : {}'.format(data_path))
    rt_path = os.path.dirname(data_path)
    g_path = os.path.join(rt_path, 'gallerySet')
    p_path = os.path.join(rt_path, 'probeSet')
    if not os.path.exists(g_path):
        os.mkdir(g_path)
    if not os.path.exists(p_path):
        os.mkdir(p_path)
    for sub_folder in os.listdir(data_path):
        abs_folder = os.path.join(data_path, sub_folder)
        sub_g_path = os.path.join(g_path, sub_folder)
        sub_p_path = os.path.join(p_path, sub_folder)
        pics = os.listdir(abs_folder)
        if len(pics) == 0:
            continue
        elif len(pics) == 1:
            if not os.path.exists(sub_g_path):
                os.mkdir(sub_g_path)
        else:
            if not os.path.exists(sub_g_path):
                os.mkdir(sub_g_path)
            if not os.path.exists(sub_p_path):
                os.mkdir(sub_p_path)
        g_pic = pics[0]
        p_pics = pics[1:]
        abs_g_pic = os.path.join(abs_folder, g_pic)
        assert os.path.exists(abs_g_pic)
        shutil.copy(abs_g_pic, sub_g_path)
        for p_pic in p_pics:
            abs_p_pic = os.path.join(abs_folder, p_pic)
            shutil.copy(abs_p_pic, sub_p_path)



if __name__=='__main__':
    data_path = '/Users/royalli/Dataset/LFW/lfw-org/lfw'
    processor(data_path)
