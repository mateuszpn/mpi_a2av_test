#include "mpi.h"

#include "CL/sycl.hpp"

#include <assert.h>
#include <iostream>
#include <iterator>
#include <vector>

#define USE_SYCL

int comm_size_, comm_rank_;

template <typename T>
std::vector<T> generate_random(std::size_t n, std::size_t bound = 25) {
  std::vector<T> v;
  v.reserve(n);

  for (std::size_t i = 0; i < n; i++) {
    auto r = lrand48() % bound;
    v.push_back(r);
  }
  return v;
}

#ifdef USE_SYCL
template <typename T>
using V = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;
#else
template <typename T> using V = std::vector<T>;
#endif

template <typename T>
void alltoallv(const V<T> &sendbuf, const std::vector<int> &sendcnt,
               const std::vector<int> &senddsp, V<T> &recvbuf,
               const std::vector<int> &recvcnt,
               const std::vector<int> &recvdsp) {

  assert(std::size(sendcnt) == comm_size_ && "Bad vector size");
  assert(std::size(senddsp) == comm_size_ && "Bad vector size");
  assert(std::size(recvcnt) == comm_size_ && "Bad vector size");
  assert(std::size(recvdsp) == comm_size_ && "Bad vector size");

  std::cout << "MPI_Alltoallv()" << std::endl;

  MPI_Alltoallv(std::data(sendbuf), std::data(sendcnt), std::data(senddsp),
                MPI_BYTE, std::data(recvbuf), std::data(recvcnt),
                std::data(recvdsp), MPI_BYTE, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  using T = int;
  std::cout << "main()" << std::endl;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank_);

  sycl::queue q(sycl::gpu_selector_v);
  std::cout << comm_rank_ << ": Running on: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // int size = 135167;
  const int fullsize = 8000000;
  int size = fullsize / comm_size_;

  std::cout << comm_rank_ << ": MPI Initialized, size " << comm_size_
            << std::endl;
  std::cout << comm_rank_ << ": vector size " << size << std::endl;

  std::vector<int> sendcnt(comm_size_), senddsp(comm_size_),
      recvcnt(comm_size_), recvdsp(comm_size_);

  std::size_t recvsize = 0;

  for (int i = 0; i < comm_size_; i++) {
    sendcnt[i] = recvcnt[i] = size;
    senddsp[i] = recvdsp[i] = i * size;
    recvsize += recvcnt[i];
  }

  // sendcnt = {size, size, size, size};
  // senddsp = {0, size, 2 * size, 3 * size};
  // recvcnt = {size, size, size, size};
  // recvdsp = {0, size, 2 * size, 3 * size};

  // std::size_t recvsize = recvcnt[0] + recvcnt[1] + recvcnt[2] + recvcnt[3];

#ifdef USE_SYCL
  std::cout << "With SYCL alloc" << std::endl;
  sycl::usm_allocator<T, sycl::usm::alloc::shared> alloc(q);
  std::vector<T, decltype(alloc)> vs(size, alloc);
  std::vector<T, decltype(alloc)> vr(recvsize, alloc);
#else
  std::cout << "With default alloc" << std::endl;
  std::vector<T> vs = generate_random<T>(size, 1000);
  std::vector<T> vr(recvsize);
#endif

  std::cout << "alltoallv()" << std::endl;
  alltoallv(vs, sendcnt, senddsp, vr, recvcnt, recvdsp);
  std::cout << "end" << std::endl;
  MPI_Finalize();
}
