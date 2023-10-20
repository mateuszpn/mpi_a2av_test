#include "mpi.h"

#include "CL/sycl.hpp"

#include <assert.h>
#include <iostream>
#include <iterator>
#include <vector>

#define USM_SHARED_MEM

#define SIZE 1000

#ifdef USM_SHARED_MEM
template <typename T>
using Alloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
#else
template <typename T>
using Alloc = sycl::usm_allocator<T, sycl::usm::alloc::host>;
#endif

template <typename T> using V = std::vector<T, Alloc<T>>;

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

int comm_size_, comm_rank_;
int main(int argc, char **argv) {
  using T = int;
  std::cout << "main()" << std::endl;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank_);

  int size = SIZE;

  std::cout << comm_rank_ << ": MPI Initialized, size " << comm_size_
            << std::endl;
  std::cout << comm_rank_ << ": vector size " << size << std::endl;

  std::vector<int> sendcnt(comm_size_), senddsp(comm_size_),
      recvcnt(comm_size_), recvdsp(comm_size_);

  std::size_t recvsize = 0;
  for (int _i = 0; _i < comm_size_; _i++) {
    sendcnt[_i] = size;
    senddsp[_i] = _i * size;
    recvcnt[_i] = size;
    recvdsp[_i] = _i * size;
    recvsize += recvcnt[_i];
  }

  sycl::queue q;
  Alloc<T> alloc(q);

  std::vector<T, decltype(alloc)> vs(size, alloc);
  std::vector<T, decltype(alloc)> vr(recvsize, alloc);

  std::cout << "alltoallv()" << std::endl;

  alltoallv(vs, sendcnt, senddsp, vr, recvcnt, recvdsp);

  std::cout << "end" << std::endl;
  MPI_Finalize();
}
