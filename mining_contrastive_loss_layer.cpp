#include <algorithm>
#include <functional>
#include <vector>
#include <map>

#include "caffe/layers/mining_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MiningContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  int mining_ratio = this->layer_param_.contrastive_loss_param().mining_ratio();
  CHECK_GE(mining_ratio, 1);
  CHECK_LE(mining_ratio, bottom[0]->num());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  select_batches_.Reshape(bottom[0]->num(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->num(); ++i)
    select_batches_.mutable_cpu_data()[i] = Dtype(0);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void MiningContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  int mining_ratio = this->layer_param_.contrastive_loss_param().mining_ratio();
  Dtype loss(0.0);
  std::multimap<Dtype, int, std::greater<Dtype> > sorted_pos_loss;
  std::multimap<Dtype, int, std::greater<Dtype> > sorted_neg_loss;   
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      sorted_pos_loss.insert(std::multimap<Dtype, int>::value_type(dist_sq_.cpu_data()[i], i));
    } else {  // dissimilar pairs
      if (legacy_version) {
        sorted_neg_loss.insert(std::multimap<Dtype, int>::value_type(std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0)), i));
      } else {
        Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        sorted_neg_loss.insert(std::multimap<Dtype, int>::value_type(dist * dist, i));
      }
    }
  }
  int select_batches_num = bottom[0]->num()/mining_ratio;
  int selected = 0;
  typename std::multimap<Dtype, int>::iterator pos_iter = sorted_pos_loss.begin();
  typename std::multimap<Dtype, int>::iterator neg_iter = sorted_neg_loss.begin();
  for (int i = 0; i < bottom[0]->num(); ++i)
	  select_batches_.mutable_cpu_data()[i] = Dtype(0);    
  while (selected < select_batches_num) {
    if(pos_iter != sorted_pos_loss.end()) {
        loss += pos_iter->first;
        select_batches_.mutable_cpu_data()[pos_iter->second] = Dtype(1);
        selected++;
        pos_iter++;
    }
    if(selected == select_batches_num) {
        break;
    }
    if(neg_iter != sorted_neg_loss.end()) {
        loss += neg_iter->first;
        select_batches_.mutable_cpu_data()[neg_iter->second] = Dtype(1);
        selected++;
        neg_iter++;
    }    
  }    
  loss = loss / static_cast<Dtype>(selected) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MiningContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(select_batches_.num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (select_batches_.cpu_data()[j] == Dtype(0)) {
            caffe_set(channels, Dtype(0), bout + (j*channels));
            continue;
        }
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4)); //  pd/px = 1/d * (x1 - x2) , d = sqrt(sum((x1-x2)^2))
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                beta,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MiningContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(MiningContrastiveLossLayer);
REGISTER_LAYER_CLASS(MiningContrastiveLoss);

}  // namespace caffe
