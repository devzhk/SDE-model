ngc batch run --name "ml-model.fno-sde" --preempt RUNONCE --commandline  'cd /GAN-code/meta_model/SDE-model; bash ngc_scripts/train_KD.sh’ --image "nvidia/pytorch:22.01-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --result /results --workspace O7-0rdpyTiqLbdKYtM0Lkw:/GAN-code --port 6006 --port 1234 --port 8888

ngc batch run --name "ml-model.fno-sde" --preempt RUNONCE --commandline  'cd /GAN-code/meta_model/SDE-model; bash ngc_scripts/train_t5.sh' --image "nvidia/pytorch:22.01-py3" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --result /results --workspace O7-0rdpyTiqLbdKYtM0Lkw:/GAN-code --port 6006 --port 1234 --port 8888