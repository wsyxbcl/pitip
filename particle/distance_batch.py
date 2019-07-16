# Calculate all demanding distances in given directory
# and save to csv(one csv per image)
import cv2
import sys
sys.path.insert(0, '../contour')

from overlap import *
from blob import particle_blob_detection
from distance import distance_calc

parser = argparse.ArgumentParser()
parser.add_argument("image_dir", type=Path, 
                    help="Directory that contains the aimed image")
args = parser.parse_args()
working_directory = args.image_dir
saving_directory = working_directory.joinpath('output')
if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)
    
pattern = re.compile('.*?jpg$')
for filename, subdir in walker(working_directory, pattern):
    print("Openning "+filename)
    img = cv2.imread("../images/particles.jpg", 0)
    img_denoise = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    keypoints = particle_blob_detection(img_denoise)
    keypoints_coords = np.array([keypoint.pt for keypoint in keypoints])
    im_with_keypoints = cv2.drawKeypoints(img_denoise, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
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
    np.savetxt(saving_directory.joinpath(Path(filename).stem+'_dist.csv'), distances,
               delimiter=',',
               fmt='%.3e')
