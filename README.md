# mining_contrastive_loss_layer
mining_contrastive_loss_layer for Caffe
contrastive_loss_layer is mostly used in a Siamese Network. This mining_contrastive_loss_layer is used for hard sample mining.

# USAGE:
(1) Copy `mining_contrastive_loss_layer.cpp` and `mining_contrastive_loss_layer.cu` under `<caffe_root>/src/caffe/layers`. 

(2) Copy `mining_contrastive_loss_layer.hpp` under `<caffe_root>/include/caffe/layers`.

(3) Modify the `ContrastiveLossParameter` in the `caffe.proto` as the following way:
```
message ContrastiveLossParameter {
  // margin for dissimilar pair
  optional float margin = 1 [default = 1.0];
  // The first implementation of this cost did not exactly match the cost of
  // Hadsell et al 2006 -- using (margin - d^2) instead of (margin - d)^2.
  // legacy_version = false (the default) uses (margin - d)^2 as proposed in the
  // Hadsell paper. New models should probably use this version.
  // legacy_version = true uses (margin - d^2). This is kept to support /
  // reproduce existing models and results
  optional bool legacy_version = 2 [default = false];
  optional int32 mining_ratio = 3 [default = 1];
}
```
mining_ratio specifies the proportion between total batch size and mined batch size. For example, if you choose your mining_ratio equals to 2, then the net will only use the harder half samples to update the weights and bias.

mining_ratio should greater than 1 and less than batch size. If it is equal to 1, mining is not implemented, the loss is the same as contrastive loss. If it is lager than batch size, no sample is chosen for backward and all weights and bias will not update.

Note: mining will choose equal or almost(if there is not enough pos or neg in the current batch size) equal positive samples and negtive samples. 

(4) Rebuild your Caffe.  

(4) Now, use your shared_dropout_layer as follows:

```
layer {
  name: "loss"
  type: "MiningContrastiveLoss"
  bottom: "feat"
  bottom: "feat_p"
  bottom: "sim"
  top: "loss"
  contrastive_loss_param {
    margin: 1
    mining_ratio: 4
  }
}
```
