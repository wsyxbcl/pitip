import cv2
import numpy as np
import sys
sys.path.insert(0, '../contour')


from blob import particle_blob_detection
from distance import distance_calc
from overlap import *


def find_best_blob(img):
    """
    Interactivly find proper parameter for denoising and blob detection.
    In order to make blob.py clean, this function is defined in the workflow script. 
    """
    # Create a black image, a window
    # img = cv2.imread("../images/particles.jpg", 0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('image', 500, 500)
    # cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # create trackbars for denoising
    cv2.createTrackbar('NlMean', 'image', 0, 1, lambda x: None) # fastNlMeansDenoising
    cv2.createTrackbar('h', 'image', 10, 100, lambda x: None)
    cv2.createTrackbar('tws', 'image', 3, 49, lambda x: None) # 0.5(templateWindowSize-1)
    cv2.createTrackbar('sws', 'image', 10, 49, lambda x: None) # 0.5(searchWindowSize-1)

    cv2.createTrackbar('Median', 'image', 0, 1, lambda x: None) # medianBlur
    cv2.createTrackbar('k', 'image', 2, 49, lambda x: None) # 0.5(ksize-1)

    # trackbars for blob detection
    cv2.createTrackbar('Blob', 'image', 0, 1, lambda x: None)
    cv2.createTrackbar('min_thold', 'image', 10, 255, lambda x: None)
    cv2.createTrackbar('max_thold', 'image', 200, 255, lambda x: None)
    cv2.createTrackbar('step', 'image', 5, 50, lambda x: None)
    cv2.createTrackbar('min_area', 'image', 5, 100, lambda x: None)
    # cv2.imshow('image', img)

    print("ATTENTION: After parameter setting, press ESC to quit the openCV window.")
    print("(Do not click the close button of the window)")
    # Choose denoising method
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Get parameters from trackbars
        nlmean_denoise = cv2.getTrackbarPos('NlMean', 'image')
        try:
            if nlmean_denoise == -1:
                raise ValueError('Getting -1 from Trackbar')
        except ValueError:
            print("Use ESC to finish the function!")
            cv2.destroyAllWindows()
            raise
        median_denoise = cv2.getTrackbarPos('Median', 'image')
        h_mean = cv2.getTrackbarPos('h', 'image')
        tsize_mean = 2 * cv2.getTrackbarPos('tws', 'image') + 1
        ssize_mean = 2 * cv2.getTrackbarPos('sws', 'image') + 1
        ksize_median = 2 * cv2.getTrackbarPos('k', 'image') + 1
        min_thold = cv2.getTrackbarPos('min_thold', 'image')
        max_thold = cv2.getTrackbarPos('max_thold', 'image')
        step = cv2.getTrackbarPos('step', 'image')
        min_area =  cv2.getTrackbarPos('min_area', 'image')

        if nlmean_denoise and median_denoise:
            img_denoised_mean = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
            img_denoised = cv2.medianBlur(img_denoised_mean, ksize_median)
        elif median_denoise:
            img_denoised = cv2.medianBlur(img, ksize_median)
        elif nlmean_denoise:
            img_denoised = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
        else:
            img_denoised = img
        # blob detection
        if cv2.getTrackbarPos('Blob', 'image'):
            keypoints = particle_blob_detection(img_denoised, 
                                                min_area=min_area, 
                                                min_threshold=min_thold, 
                                                max_threshold=max_thold, 
                                                step_threshold=step)
            img_blob = cv2.drawKeypoints(img_denoised, keypoints, np.array([]), (0,0,255), 
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            img_blob = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_GRAY2BGR)
        img_show = np.hstack((img_denoised, img_blob))
        cv2.imshow('image', img_show)

    cv2.destroyAllWindows()
    return (nlmean_denoise, h_mean, tsize_mean, ssize_mean, median_denoise, 
            ksize_median, min_thold, max_thold, step, min_area)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path, 
                        help="Directory that contains the aimed image")
    args = parser.parse_args()
    working_directory = args.image_dir
    saving_directory = working_directory.joinpath('output')
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
        
    pattern = re.compile('.*?jpeg$')
    find_param = 0
    for filename, subdir in walker(working_directory, pattern):
        # Ignore images in the output directory
        if Path(subdir) == saving_directory:
            continue
        print("Openning "+filename)
        img = cv2.imread(str(Path(subdir).joinpath(filename)), 0)

        # Use first image to set parameters
        if not find_param:
            print("Setting parameters for denoise and blob detection.")
            (nlmean_denoise, h_mean, tsize_mean, ssize_mean, 
             median_denoise, ksize_median, 
             min_thold, max_thold, step, min_area) = find_best_blob(img)
            find_param = 1
            print("Parameters all set")
            print()

        # Denoise
        if nlmean_denoise and median_denoise:
            img_denoised_mean = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
            img_denoised = cv2.medianBlur(img_denoised_mean, ksize_median)
        elif median_denoise:
            img_denoised = cv2.medianBlur(img, ksize_median)
        elif nlmean_denoise:
            img_denoised = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
        else:
            img_denoised = img
        # Blob detection
        keypoints = particle_blob_detection(img_denoised, min_area=min_area, 
                                            min_threshold=min_thold, max_threshold=max_thold,
                                            step_threshold=step)
        keypoints_coords = np.array([keypoint.pt for keypoint in keypoints])
        im_with_keypoints = cv2.drawKeypoints(img_denoised, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Plot & save figure
        fig = plt.figure(tight_layout=True)
        gs = matplotlib.gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])  
        ax2 = plt.subplot(gs[0, 1])
        ax1.title.set_text('Original image')
        ax1.imshow(img, cmap = 'gray')
        ax2.title.set_text('Denoise & Blob_detection')
        ax2.imshow(im_with_keypoints)
        imgname = os.path.splitext(filename)[0]
        plt.savefig(saving_directory.joinpath(Path(filename).stem+'_blob.png'), 
                    dpi=300)
        plt.clf()
        distances = distance_calc(keypoints_coords)
        # Saving distance matrix
        np.savetxt(saving_directory.joinpath(Path(filename).stem+'_dist.csv'), distances,
                delimiter=',',
                fmt='%.3e')