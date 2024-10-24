#include <iostream>
#include <algorithm>
#include <cassert>
#include <chrono>

using namespace std::chrono;
using namespace std;

template <typename MatrixType> struct matrix_desc {
 typedef typename MatrixType::index_type index_type;
 typedef typename MatrixType::value_type value_type;
 static index_type min_row(MatrixType const& A) {
  return A.min_row();
 }
 static index_type max_row(MatrixType const& A) {
  return A.max_row();
 }
 static index_type min_column(MatrixType const& A) {
  return A.min_column();
 }
 static index_type max_column(MatrixType const& A) {
  return A.max_column();
 }
 static value_type& element(MatrixType& A, index_type i, index_type k) {
  return A(i, k);
 }
 static value_type element(MatrixType const& A, index_type i, index_type k) {
  return A(i, k);
 }
};

template <typename T, size_t rows, size_t columns> struct matrix_desc <T[rows][columns]> {
 typedef size_t index_type;
 typedef T value_type;
 static index_type min_row(T const (&)[rows][columns]) {
  return 0;
 }
 static index_type max_row(T const (&)[rows][columns]) {
  return rows - 1;
 }
 static index_type min_column(T const (&)[rows][columns]) {
  return 0;
 }
 static index_type max_column(T const (&)[rows][columns]) {
  return columns - 1;
 }
 static value_type& element(T(&A)[rows][columns], index_type i, index_type k) {
  return A[i][k];
 }
 static value_type element(T const (&A)[rows][columns], index_type i, index_type k) {
  return A[i][k];
 }
};

template<typename MatrixType> void swap_rows(MatrixType& A,
 typename matrix_desc<MatrixType>::index_type i,
 typename matrix_desc<MatrixType>::index_type k) {
 matrix_desc<MatrixType> mt;
 typedef typename matrix_desc<MatrixType>::index_type index_type;
 assert(mt.min_row(A) <= i);
 assert(i <= mt.max_row(A));
 assert(mt.min_row(A) <= k);
 assert(k <= mt.max_row(A));
 for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
  swap(mt.element(A, i, col), mt.element(A, k, col));
}

// Деление строки i матрицы A на v
template <typename MatrixType> void divide_row(MatrixType& A,
 typename matrix_desc<MatrixType>::index_type i,
 typename matrix_desc<MatrixType>::value_type v) {
 matrix_desc<MatrixType> mt;
 typedef typename matrix_desc<MatrixType>::index_type index_type;
 assert(mt.min_row(A) <= i);
 assert(i <= mt.max_row(A));
 assert(v != 0);
 for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
  mt.element(A, i, col) /= v;
}

// В матрице A добавить v раз строку k к строке i
template<typename MatrixType>
void add_multiple_row(MatrixType& A,
 typename matrix_desc<MatrixType>::index_type i,
 typename matrix_desc<MatrixType>::index_type k,
 typename matrix_desc<MatrixType>::value_type v) {
 matrix_desc<MatrixType> mt;
 typedef typename matrix_desc<MatrixType>::index_type index_type;
 assert(mt.min_row(A) <= i);
 assert(i <= mt.max_row(A));
 assert(mt.min_row(A) <= k);
 assert(k <= mt.max_row(A));
 #pragma omp simd
 for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
  mt.element(A, i, col) += v * mt.element(A, k, col);
}

// Преобразовать A к ступенчатому виду
template<typename MatrixType> void to_canonical_rows_form(MatrixType& A) {
 matrix_desc<MatrixType> mt;
 typedef typename matrix_desc<MatrixType>::index_type index_type;
 index_type lead = mt.min_row(A);
 for (index_type row = mt.min_row(A); row <= mt.max_row(A); ++row) {
  if (lead > mt.max_column(A)) return;
  index_type i = row;
  while (mt.element(A, i, lead) == 0) {
   i++;
   if (i > mt.max_row(A)) {
    i = row;
    lead++;
    if (lead > mt.max_column(A)) return;
   }
  }
  swap_rows(A, i, row);
  divide_row(A, row, mt.element(A, row, lead));
  #pragma omp simd
  for (i = mt.min_row(A); i <= mt.max_row(A); i++) {
   if (i != row) add_multiple_row(A, i, row, -mt.element(A, i, lead));
  }
 }
}

int main() {
 const int N = 3, M = 5;
 alignas(32) double A[N][M] = {
  { 3,2,2,3,1 },
  { 6,4,4,6,2 },
  { 9,6,6,9,1 }
 };
 auto start = high_resolution_clock::now();
 to_canonical_rows_form(A);
 auto stop = high_resolution_clock::now();
 for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) cout << A[i][j] << '\t';
  cout << endl;
 }
 auto duration = duration_cast<nanoseconds>(stop - start);
 std::cout << "Time taken: " << duration.count() << " nanoseconds\n";
 return 0;
}
