import os
import sys
import cv2
import numpy as np
import util_cal
import util_pipe
import util_lane
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from moviepy.editor import VideoFileClip
    
def get_processor(nbins=10):
    bins = nbins
    l_params = deque(maxlen=bins)
    r_params = deque(maxlen=bins)
    l_radius = deque(maxlen=bins)
    r_radius = deque(maxlen=bins)
    weights = np.arange(1,bins+1)/bins
    def process_image(image):
        img, img_size = util_cal.get_undistorted_image(image, mtx, dist)
        src, dst = util_cal.get_transform_points(img_size)
        bird, M, Minv = util_cal.warp_image(img,src,dst,(img_size[1],img_size[0]))
        # util_cal.plt_n([new_img,img,bird],['original','undistorted','birds-eye'])
        #bird = util_pipe.histogram_equalize(bird)
        #img_bin = util_pipe.color_threshold(bird,s_thresh=(200,255),v_thresh=(200,255))
        #img_bin = util_pipe.pipeline_edge(bird,s_thresh=(150,255),g_thresh=(150,255))
        img_bin = util_pipe.pipeline_YW(bird)

        if slide_window_only or len(l_params)==0:  
            left_fit,right_fit,left_curverad,right_curverad,out_img = util_lane.slide_window_fit(img_bin) 
        else:
            left_fit,right_fit,left_curverad,right_curverad,out_img = util_lane.using_prev_fit(img_bin,   
                                                                np.average(l_params,0,weights[-len(l_params):]),
                                                                np.average(r_params,0,weights[-len(l_params):]))
        
        l_params.append(left_fit)
        r_params.append(right_fit)
        l_radius.append(left_curverad)
        r_radius.append(right_curverad)

        img_out = util_lane.mapping_fit_lane(bird, img_bin, img,  
                        np.average(l_params,0,weights[-len(l_params):]),
                        np.average(r_params,0,weights[-len(l_params):]),
                        np.average(l_radius,0,weights[-len(l_params):]),
                        np.average(r_radius,0,weights[-len(l_params):]), Minv)

        diagnostic_output = True
        if diagnostic_output:
            # put together multi-view output
            diag_img = np.zeros((720,1280,3), dtype=np.uint8)
            
            # original output (top left)
            diag_img[0:720,0:1280,:] = cv2.resize(img_out,(1280,720))
            
            resized_out_img = None
            if out_img is not None:
                if out_img.shape[0]>0 and out_img.shape[1]>0:
                    resized_out_img = cv2.resize(out_img,(320,180)) # img_bin,(640,360))
            else:
                img_bin = np.dstack((img_bin*255, img_bin*255, img_bin*255))
                resized_out_img = cv2.resize(img_bin,(320,180))

            if resized_out_img is not None:
                diag_img[0:180,960:1280, :] = resized_out_img

            # resized_bird = cv2.resize(bird,(320,180))
            #diag_img[0:180,320:640, :] = resized_bird

            img_out = diag_img

        return img_out

    return process_image

def do_image(img):
    return (get_processor(1))(img)

def do_image_file(fin, fout, save=True):
    img = mpimg.imread(fin)
    img_out = do_image(img)
    if save:
        mpimg.imsave(fout, img_out)
        print('Save output image to ',fout, ' ...')
    return img_out

"""
Main function
"""
#args = set_args()
#dir = args.data_dir

slide_window_only = False

mtx, dist = util_cal.get_undistorted_params('calibration_pickle.p')
def main():
    #my_clip.write_gif('test.gif', fps=12)

    if not os.path.isdir("output_images"):
        os.mkdir("output_images")

    pj_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip('project_video.mp4') #.subclip(22,26)
    pj_clip = clip1.fl_image(get_processor(15)) # process_image)
    pj_clip.write_videofile(pj_output, audio=False)

    ch_output = 'output_images/challenge_video.mp4'
    clip2 = VideoFileClip("./challenge_video.mp4")
    ch_clip = clip2.fl_image(get_processor(15)) 
    ch_clip.write_videofile(ch_output, audio=False)

    # harder_output = 'output_images/harder_challenge_video.mp4'
    # clip3 = VideoFileClip("./harder_challenge_video.mp4")
    # harder_clip = clip3.fl_image(get_processor(5)) 
    # harder_clip.write_videofile(harder_output, audio=False)


if __name__ == '__main__':
    n = len(sys.argv)
    if n>1:
        fin = sys.argv[1]
        fout = "out.png" if n==2 else sys.argv[2]
        do_image_file(fin, fout)
        
    else:
        main()
