import matplotlib.pyplot as plt
from utils.path_utils import get_results_dir

def compare(categories, num_images):

    num_cols = len(categories)
    num_rows = num_images
    
    plt.figure(figsize=(3*num_cols, 4))
    for col_idx, category in enumerate(categories):
        for row_idx in range(num_rows):
            histogram = get_results_dir() / f"histogram_{category}_{row_idx}.png"
            img = plt.imread(histogram)
                
            plt.subplot(num_rows, num_cols, 1 + col_idx + num_cols*row_idx)
            plt.imshow(img)
            plt.axis("off")
            
    plt.subplots_adjust(left=0.00,
                        bottom=0.01, 
                        right=1.0, 
                        top=0.99, 
                        wspace=0.01, 
                        hspace=0.01)
    plt.savefig(get_results_dir() / "compare.png")
