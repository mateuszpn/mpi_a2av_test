#include "mpi.h"

#include <assert.h>
#include <iterator>
#include <ranges>
#include <vector>

int comm_size_;

template <typename valT>
void alltoallv(std::vector<valT> &sendbuf,
               const std::vector<std::size_t> &sendcnt,
               const std::vector<std::size_t> &senddsp,
               std::vector<valT> &recvbuf,
               const std::vector<std::size_t> &recvcnt,
               const std::vector<std::size_t> &recvdsp) {

  assert(std::size(sendcnt) == comm_size_);
  assert(std::size(senddsp) == comm_size_);
  assert(std::size(recvcnt) == comm_size_);
  assert(std::size(recvdsp) == comm_size_);

  std::vector<int> _sendcnt(comm_size_);
  std::vector<int> _senddsp(comm_size_);
  std::vector<int> _recvcnt(comm_size_);
  std::vector<int> _recvdsp(comm_size_);

  for (int i = 0; i < comm_size_; i++) {
    _sendcnt[i] = sendcnt[i] * sizeof(valT);
    _senddsp[i] = senddsp[i] * sizeof(valT);
    _recvcnt[i] = recvcnt[i] * sizeof(valT);
    _recvdsp[i] = recvcnt[i] * sizeof(valT);
  }

  MPI_Alltoallv(std::data(sendbuf), std::data(_sendcnt), std::data(_senddsp),
                MPI_BYTE, std::data(recvbuf), std::data(_recvcnt),
                std::data(_recvdsp), MPI_BYTE, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  using T = int;
  constexpr std::size_t size = 2000000;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);

  std::vector<T> v(size / comm_size_);
  std::vector<int> sendcnt({125146, 124922, 125145, 124787});
  std::vector<int> senddsp({0, 125146, 250068, 375213});
  std::vector<int> recvcnt({125025, 124785, 125145, 125296});
  std::vector<int> recvdsp({0, 125025, 249810, 374955});
}