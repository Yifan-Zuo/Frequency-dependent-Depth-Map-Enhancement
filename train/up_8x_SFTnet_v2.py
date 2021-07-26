'''
Created on 07Sep,2019

@author: yizuo
'''
import tensorflow as tf
import sft_basic_bks_v2 as sbb
import h5py

up_factor=8
batch_sz=32
height=128
width=128

#setting input size and training data addr
epo_range=60
train_h5F_addr="/media/yifan/Data/training_data/gdsr_train_data/8x_data/shuffle_version/8x_training_data.h5"
total_pat=220500
total_val_pat=24500
LR_height=height/up_factor
LR_width=width/up_factor
batch_total=total_pat/batch_sz
val_batch_total=total_val_pat/batch_sz
HR_patch_size=[height,width]
HR_batch_dims=(batch_sz,height,width,1)
LR_batch_dims=(batch_sz,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#network construction
inten_feat=sbb.inten_FE(HR_inten_batch_input)
dep_feat=sbb.dep_FE(LR_depth_batch_input)
inten_feat_down=sbb.inten_FDown(inten_feat,2,2)
dep_feat_up=sbb.deconv_Prelu_block(dep_feat,[5,5,32,32],[1,2,2,1],False)
dep_feat_base=dep_feat_up
for ind in [2,1,0]:
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
#loss=tf.reduce_mean(tf.squared_difference(HR_gen_dep,HR_depth_batch_input))
loss=tf.reduce_mean(tf.abs(HR_gen_dep-HR_depth_batch_input))
#loss=tf.reduce_mean(tf.sqrt(tf.squared_difference(HR_gen_dep,HR_depth_batch_input)+1e-3))
train_op_small = tf.train.AdamOptimizer(1e-5).minimize(loss)
train_op_large = tf.train.AdamOptimizer(1e-4).minimize(loss)
saver_full=tf.train.Saver(max_to_keep=1200)
model_ind=0
init_op=tf.global_variables_initializer()

#begin comp_gen training
with h5py.File(train_h5F_addr,"r") as train_file:
    with tf.Session() as sess:
        sess.run(init_op)
        #saver_full.restore(sess, "/media/yifan/Data/trained_models/sft_depup_net/noise_free/l1loss/8x/best_model/full_model/8x_nf_cgsft_model.ckpt-119")
        for epo in range(epo_range):
            if epo<20:
                train_op=train_op_large
            else:
                train_op=train_op_small
            for ind in range(batch_total):
                gen_pat_ind_range=range(ind*batch_sz,(ind+1)*batch_sz,1)
                gen_inten_bat,gen_gth_dep_bat,gen_LR_dep_bat=sbb.reading_data(train_file, gen_pat_ind_range, HR_batch_dims, LR_batch_dims)
                sess.run(train_op,feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                if (ind+1)%957==0:
                    mae_loss=loss.eval(feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                    print("step %d, training loss %g"%(ind, mae_loss))
                if (ind+1)%3828==0:
                    save_path=saver_full.save(sess,"/media/yifan/Data/trained_models/sft_depup_net/noise_free/l1loss/8x/best_model/full_model3/8x_nf_cgsft_model.ckpt",global_step=model_ind)
                    print("Model saved in file: %s" % save_path)
                    val_mae_loss=0
                    for val_ind in range(val_batch_total):
                        gen_pat_ind_range=range(total_pat+val_ind*batch_sz,total_pat+(val_ind+1)*batch_sz,1)
                        gen_inten_bat,gen_gth_dep_bat,gen_LR_dep_bat=bnb.reading_data(train_file, gen_pat_ind_range, HR_batch_dims, LR_batch_dims)
                        mae_loss=loss.eval(feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                        val_mae_loss=val_mae_loss+mae_loss
                    val_mae_loss=val_mae_loss/val_batch_total
                    print("model %d, validation loss %g"%(model_ind, val_mae_loss))
                    model_ind=model_ind+1
