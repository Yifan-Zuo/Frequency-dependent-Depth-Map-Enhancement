'''
Created on 16Sep,2019

@author: yizuo
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import sft_basic_bks_v2 as sbb

up_factor=2
data_name="art"
#read test HR inten img
color_art = Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/color_"+data_name+"_croped.png")#####for noise-free midd2006
#color_art = Image.open("/media/yifan/Data/test_data/noisy/table2/_gth/"+data_name+"_color.png")####mid 2006
#color_art = Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/color_crop.png")####mid 2001
inten_art =color_art.convert("L")
np_inten=np.asarray(inten_art)
val_inten=np_inten.astype(np.float32)/255.0

#read test HR dep img
gth_dep_art=Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"_croped.png")#####for noise-free midd2006
#gth_dep_art=Image.open("/media/yifan/Data/test_data/noisy/table2/_gth/"+data_name+"_big.png")###mid 2006
#gth_dep_art=Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/depth_crop.png")###mid 2001
np_gth_dep=np.asarray(gth_dep_art)
val_gth_dep=np_gth_dep.astype(np.float32)/255.0

#read test LR dep img
LR_dep_art=Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"2x_bicubic.png")#####for noise-free midd2006
#LR_dep_art=Image.open("/media/yifan/Data/test_data/noisy/table2/_input/"+data_name+"_big/depth_1_n.png")###mid 2006
#LR_dep_art=Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/input/"+data_name+"/2_x_dep.png")###mid 2001
np_LR_dep=np.asarray(LR_dep_art)
val_LR_dep=np_LR_dep.astype(np.float32)/255.0

height=val_inten.shape[0]
width=val_inten.shape[1]
LR_height=height/up_factor
LR_width=width/up_factor
val_inten=val_inten.reshape((1,height,width,1))
val_gth_dep=val_gth_dep.reshape(1,height,width,1)
val_LR_dep=val_LR_dep.reshape(1,LR_height,LR_width,1)

#setting input size and training data addr
HR_patch_size=[height,width]
HR_batch_dims=(1,height,width,1)
LR_batch_dims=(1,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#gen_network construction
inten_feat=sbb.inten_FE(HR_inten_batch_input)
dep_feat=sbb.dep_FE(LR_depth_batch_input)
#inten_feat_down=sbb.inten_FDown(inten_feat,2,2)
inten_feat_down=[inten_feat]
dep_feat_up=sbb.deconv_Prelu_block(dep_feat,[5,5,32,32],[1,2,2,1],False)
dep_feat_base=dep_feat_up
for ind in [0]:
    for iter_ind in range(2):
        ref_inten=sbb.sft_layer(inten_feat_down[ind],dep_feat_up)
        dep_feat_up=sbb.feat_fusion(ref_inten,dep_feat_up)
        if iter_ind==0:
            mul_dep_feat=dep_feat_up
        else:
            mul_dep_feat=tf.concat([mul_dep_feat,dep_feat_up],3)
    mul_dep_feat=sbb.conv_Prelu_block(mul_dep_feat, [1,1,32*(iter_ind+1),32], [1,1,1,1], False)
    if ind:
        dep_feat_up=sbb.dep_FUp(mul_dep_feat,dep_feat_base)
        dep_feat_base=dep_feat_up
HR_gen_dep=sbb.recon(coar_inter_dep_batch,mul_dep_feat)

#define loss for gen
saver_full=tf.train.Saver()

#begin comp_gen testing
with tf.Session() as sess:
    model_path="/media/yifan/Data/trained_models/sft_depup_net/noise_free/l1loss/2x/full_model/2x_nf_cgsft_model.ckpt-117"
    saver_full.restore(sess, model_path)
    ten_fets=sess.run(HR_gen_dep,feed_dict={HR_inten_batch_input:val_inten,HR_depth_batch_input:val_gth_dep,LR_depth_batch_input:val_LR_dep})
    final_array=ten_fets*255.0+0.5
    final_array[final_array>255]=255.0
    final_array[final_array<0]=0.0
    final_array=final_array.astype(np.uint8).reshape((height,width))
    result_img=Image.fromarray(final_array)
    #result_img.show()
    result_img.save("/media/yifan/Data/trained_models/sft_depup_net/noise_free/l1loss/2x/result/"+data_name+"2x_result.png")
    ######################computing rmse
    final_array=final_array.astype(np.double)
    np_gth_dep=np_gth_dep.astype(np.double)
    print("model %d:"%ind)
    print(np.sqrt(((final_array-np_gth_dep)**2).mean()))
    print((np.absolute(final_array-np_gth_dep)).mean())
    #print("below are evaluated on 1080*1320")
    #print(np.sqrt(((final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])**2).mean()))
    #print((np.absolute(final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])).mean())
    #######################
