import os
import tifffile

half_p_size = 112
patch_size = 224


def get_slide_tissue_mask(slide_name, path=None):
    tissue_mask_fname = os.path.join(path, 'tissue_mask_' + slide_name)
    if not os.path.isfile(tissue_mask_fname):
        print('Error: file {} does not exist'.format(tissue_mask_fname))
    else:
        tissue_mask = tifffile.imread(tissue_mask_fname)
        return tissue_mask



def return_ratios(ci, cj, tissue_mask):
    # Here we're getting the column pixel where the patch starts
    start_col = ci - half_p_size
    # The column pixel where the patch ends
    end_col = ci + half_p_size
    # start and end row follow the same idea
    start_row = cj - half_p_size
    end_row = cj + half_p_size

    mask_max = patch_size * patch_size * 255  # this is what a mask that is all one class
    patch_tissue_mask_sum = tissue_mask[start_row:end_row, start_col:end_col].sum()
    patch_tissue_ratio = patch_tissue_mask_sum / mask_max
    return patch_tissue_ratio

def get_indices_lowerlvl(current_center_cords,downsample_current, downsample_factor_lower_level):
    # level 0 center cords
    center_level_zero_cords = (current_center_cords[0] * downsample_current,
                               current_center_cords[1] * downsample_current)  # this puts the cords in level 0
    # lower (next level) center cords
    center_level_lower_cords = (center_level_zero_cords[0] / downsample_factor_lower_level,
                                center_level_zero_cords[1] / downsample_factor_lower_level)

    return (center_level_lower_cords[0],center_level_lower_cords[1])