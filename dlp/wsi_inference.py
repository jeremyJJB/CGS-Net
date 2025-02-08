import glob
import os.path
# import tifffile
import tiffslide
import torch
from PIL import Image
from einops import rearrange
import wsi_inference_utils as ws_utils
import numpy as np
from models.model import ModelMaker
from empatches import EMPatches
from datetime import datetime
emp = EMPatches()
from multiprocessing import Pool
# from functools import partial


def wsi_inference(testwsi_fp):
    torch.set_num_threads(7)
    # pytorch will automatically use all avialble threads, so when multithreading model inference. we need to make sure that each thread
    # only uses some many cpus otherwise the threads will fight over the cpus.
    model_obj = ModelMaker(model_params=model_options)
    model = model_obj.make_model()
    model_obj.print_model_params(model=model)
    model.to('cpu')
    model.eval()
    # need to update this code if doing single input model
    test_transform = model_obj.make_val_transform()
    test_transform.add_targets({'image1': 'image'})
    print("model made", flush=True)
    # print("in thread call ", flush=True)
    # print(testwsi_fp, flush=True)
    # print(type(testwsi_fp), flush=True)
    slidename = os.path.split(testwsi_fp)[-1]
    print(f"working on slide {slidename}",flush=True)
    slide = tiffslide.open_slide(testwsi_fp)  # this opens the slide but we need to close it
    slide_map = np.float32(slide.get_thumbnail(slide.level_dimensions[level]))
    # slide_map = np.float32(slide.get_thumbnail(slide.level_dimensions[level]))[:4000,:10000,:] # rgb slide TODO:CHANGE
    tissue_slide_mask = ws_utils.get_slide_tissue_mask(slidename, slide_mask_path)

    img_patches, indices = emp.extract_patches(slide_map, patchsize=224, overlap=0.80, stride=56)
    print("56 is new stride ", flush=True)
    print("0.80 is new overlap ", flush=True)

    # we are doing a pass on the indices and removing any indicies (for a patch) that conatine less than x% tissue
    # in other words we are removing the background patches from the test slides
    print(f"number of patches from {slidename}",len(img_patches),flush=True)
    print(f"num indices {slidename}", len(indices), flush=True)
    copy_index_indices = indices.copy()
    # num_patches_removed = 0
    # for p_index in copy_index_indices: # note when popping repeatedly very quickly python was not updating the list correclty, we needed to loop over a copy of the pure list
    #     # imagine we are at index 20, we then pop index 20, the next index we go to in the for loop is index 21 but the list has shifted since we poped and hence
    #     # we never hit the og index 21 (which got shifted to 20). so skip parts
    #     ci, cj = p_index[2]+112, p_index[0] + 112
    #     tissue_ratio = ws_utils.return_ratios(ci=ci,cj=cj,tissue_mask=tissue_slide_mask)
    #     # print(tissue_ratio, p_index,flush=True)
    #     if tissue_ratio < 0.10:
    #         num_patches_removed += 1
    #         img_patches.pop(indices.index(p_index))
    #         indices.pop(indices.index(p_index)) # remove by value, doing it like this to ensure removing the element at the correct index location


    # print('num after data cleaning',len(img_patches),flush=True)
    # print('num after data cleaning indices',len(indices),flush=True)

    # print('num removed',num_patches_removed,flush=True)


    slide_map_lv3 = np.float32(slide.get_thumbnail(slide.level_dimensions[level+1]))  # rgb slide
    slide.close()
    del slide_map

    lv3_prep = [(x+112,y+112) for y,yoff,x,xoff in indices]
    lv3_indices = [ws_utils.get_indices_lowerlvl(current_center_cords=cords, downsample_current=resolution_levels[level], downsample_factor_lower_level=resolution_levels[level+1]) for cords in lv3_prep]
    # print(indices[0],flush=True)
    output_patches = []
    border_width = 20
    with torch.no_grad():
        for idx,lvl2_patch in enumerate(img_patches):

            ci, cj = copy_index_indices[idx][2] + 112, copy_index_indices[idx][0] + 112
            tissue_ratio = ws_utils.return_ratios(ci=ci, cj=cj, tissue_mask=tissue_slide_mask)
            # print(tissue_ratio, p_index,flush=True)
            if tissue_ratio < 0.05:
                myzeros = np.zeros((224,224))
                output_patches.append(myzeros)
                continue

            # lvl2_tensor = torch.from_numpy()
            # print(indices[0])
            # print(lv3_indices[idx][0]-112)
            # print(lv3_indices[idx][0]+112)
            # print(lv3_indices[idx][1]-112)
            # print(lv3_indices[idx][1]+112)
            lvl3_patch = slide_map_lv3[int(lv3_indices[idx][1]-112):int(lv3_indices[idx][1]+112), int(lv3_indices[idx][0]-112):int(lv3_indices[idx][0]+112)]
            lv2_np = np.array(lvl2_patch)
            empty_mask = np.zeros(lv2_np.shape) #the error could be below, maybe there is a bad patch from emp patches which then gets passed as non to augmentions hence the error
            augmentations = test_transform(image=lv2_np, image1=lvl3_patch, mask=empty_mask)
            image_detail = augmentations["image"]
            image_context = augmentations["image1"]
            image_context = rearrange(image_context, 'c h w -> 1 c h w')
            image_detail = rearrange(image_detail, 'c h w -> 1 c h w')

            # print(image_detail.shape, flush=True)
            # print(image_context.shape, flush=True)

            # lvl3_tensor = torch.from_numpy(lvl3_patch)



            # mask_detail = augmentations["mask"]


            #TODO: I think we need to data augmentation here for these to work correctly look more closely at the test analysis
            lvl2_pred =  torch.sigmoid(model(image_detail,image_context))
            lvl2_pred = rearrange(lvl2_pred, 'b c h w -> (b c h) w')
            # print(lvl2_pred.shape, flush=True)
            # print(lvl2_pred.numpy().shape,  flush=True)
            lv2_pred_np = lvl2_pred.numpy()

            lv2_pred_np[:border_width, :] = 0 #top
            lv2_pred_np[-border_width:, :] = 0 #bottom
            lv2_pred_np[:, :border_width] = 0 #left
            lv2_pred_np[:, -border_width:] = 0 # right

            # lv2_pred_np = np.concatenate([top, bottom, left, right], axis=0)
            output_patches.append(lv2_pred_np)
            # if lvl2_pred.sum() != 0:
            #     print(lvl2_pred.sum())
            if idx % 100 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"% complete for {slidename}",len(output_patches)/len(img_patches) * 100, current_time, flush=True)
                # print(f"% complete for {slidename}",len(output_patches)/len(img_patches) * 100, current_time)

    del lv3_indices, lv3_prep, img_patches, model, slide_map_lv3, copy_index_indices
    # Merge the processed patches
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("before merge patches: ", current_time, flush=True)
    # print(indices[-1],flush=True)
    prediction_slide_mask = emp.merge_patches(output_patches, indices, mode='max')
    del output_patches, indices
    # print("sum after emp patches ", prediction_slide_mask.sum())
    # emp likes numpy arrays much better than pytorch tensors. also note that when you want all black for a prediction, you should not remove the indices from the list
    # pop, rather skip over the bad inputs in your model prediction loop, and then make your model prediction zeros. If you pop the indicies and only loop over a subset
    # then emp will not have the full dimensions of the wsi slide and you will get indexing errors. The over all flow is that you generate the patches and indicies and then loop over them
    # do things to the patches and save the outputs and then do the merge. Notice, I'm not removing anything.
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("after merge patches and before save: ", current_time, flush=True)
    # prediction_slide_mask_tiff = np.array(prediction_slide_mask, dtype=np.uint8)
    # tifffile.imwrite(f"./pred_slides/tiff_{slidename}", data=prediction_slide_mask_tiff)
    pred_pil = Image.fromarray(prediction_slide_mask)
    pred_pil.save(f"./pred_slides/{slidename}")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("after save: ", current_time, flush=True)
    return



if __name__ == '__main__':
    # note all patches used are from /home/jjuybari/data/c16_data/ml_split_refine
    # test_fp_tumor = "/home/jjuybari/data/c16_combined_slides/tumor/"
    # test_fp_normal = "/home/jjuybari/data/c16_combined_slides/normal/"

    # slide_mask_path = "/home/jjuybari/data/c16_data/sep23/mask_lv2"

    # all_tumor_slides = glob.glob(test_fp_tumor + "tumortest_*.tif")
    # all_normal_slides = glob.glob(test_fp_normal + "normaltest_*.tif")
    level = 2  # needs to be an integer
    resolution_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # slidelist = all_tumor_slides + all_normal_slides  # note that no errors were made for the test slides (only tumor patches, see utils.py
    # slidelist.sort() # see if this allows more than one instance

    ###### Remove the ones alreadly completed
    # alreadly_done = ['home/jjuybari/data/c16_combined_slides/tumor/tumortest_113.tif']

    #TODO: come up with a better way to do this

    # for slide_fp in slidelist:
    #     if alreadly_done[0] in slide_fp:
    #         slidelist.remove(slide_fp)
    
    
    slidelist = ['/home/jjuybari/data/c16_combined_slides/tumor/tumortest_010.tif',
                 '/home/jjuybari/data/c16_combined_slides/tumor/tumortest_026.tif',
                 '/home/jjuybari/data/c16_combined_slides/tumor/tumortest_104.tif',
                 '/home/jjuybari/data/c16_combined_slides/tumor/normaltest_005.tif'
                 '/home/jjuybari/data/c16_combined_slides/tumor/normaltest_070.tif'
                 '/home/jjuybari/data/c16_combined_slides/tumor/normaltest_119.tif']

    ## loading the model

    model_options = {
        "model_name": "cgs_mit_b1",
        "weights": "trainval_cgs",
        "dropout": 0.15,
        "input_img_size": 224,
        "aug_type": None,
    }

    #TODO:
    # slidelist = slidelist[:6] #TODO:CHANGE
    # slidelist = slidelist[39:78]
    # slidelist = slidelist[78:117]
    # slidelist = [slidelist[0]] # when doing just a single element from a list need additional [] to keep it in list,
    # if slicing then don't need that
    with Pool(3) as p: #TODO: CHANGE
        # p.map(partial(wsi_inference), slidelist)
        p.map(wsi_inference, slidelist)


# 2) make sure that the patch name has x,y cordinates in it and then save the top left corner x,y cord

# 3) before starting a slide init an empty array the size of the slide at the level of the prediction masks

# 4) run on inference on each patch, then do the subset np.where (the higher probability always wins allowing us
# to take advantage of overlap

# 5) then run froc code with small regions and removing small regions.

# The plan for multithreading
# epyc one node 13 slides (number of threads) 94 cores (7 cores per slide few extra), (390 GB total if doing 30 GB per slide) will do 500GB some extra in case
# do the above three times, give each node 39 slides
# so in three iterations we will have 39*3=117

# we need to fix the number of cores pontetially?

#1) slidelist[:39] p1
#2) slidelist[39:78] p2
#2) slidelist[78:117] p3

# remainder on epyc-hm, 10 slides, 4 threads 7 cores each.
#2) slidelist[117:] p5