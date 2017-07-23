#include <algorithm>
#include <functional>¡¡
#include <vector>
#include <map>

#include "caffe/layers/mining_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MiningContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  int mining_ratio = this->layer_param_.contrastive_loss_param().mining_ratio();
  Dtype loss(0.0);  
  std::multimap<Dtype, int, std::greater<Dtype> > sorted_pos_loss;
  std::multimap<Dtype, int, std::greater<Dtype> > sorted_neg_loss;
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      sorted_pos_loss.insert(std::multimap<Dtype, int>::value_type(dist_sq_.cpu_data()[i], i));
    } else {  // dissimilar pairs
      if (legacy_version) {
        sorted_neg_loss.insert(std::multimap<Dtype, int>::value_type(std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0)), i));
      } else {
        Dtype dist = std::max(margin - sqrt(dist_sq_.cpu_data()[i]),
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
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff, const Dtype* select_batches) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (select_batches[n] == Dtype(0)) {
      bottom_diff[i] = 0;  
    } else {
        if (static_cast<int>(y[n])) {  // similar pairs
        bottom_diff[i] = alpha * diff[i];
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = (margin - dist_sq[n]);
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq[n]);
            mdist = (margin - dist);
            beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
          }
          if (mdist > 0.0) {
            bottom_diff[i] = beta;
          } else {
            bottom_diff[i] = 0;
          }
        }   
    }
  }
}

template <typename Dtype>
void MiningContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(select_batches_.num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff(),
          select_batches_.gpu_data());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MiningContrastiveLossLayer);

}  // namespace caffe
