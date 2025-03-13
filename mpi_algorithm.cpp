#include <iostream>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <vector>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

const int N = 1000; // Количество строк
const int M = 1000; // Количество столбцов

template <typename MatrixType>
struct matrix_desc {
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

template <typename T, size_t rows, size_t columns>
struct matrix_desc<T[rows][columns]> {
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

template<typename MatrixType>
void swap_rows(MatrixType& A,
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
template <typename MatrixType>
void divide_row(MatrixType& A,
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

    for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
        mt.element(A, i, col) += v * mt.element(A, k, col);
}

// Преобразовать A к ступенчатому виду
template<typename MatrixType>
void to_canonical_rows_form(MatrixType& A, int rank, int size) {
    matrix_desc<MatrixType> mt;
    typedef typename matrix_desc<MatrixType>::index_type index_type;
    index_type lead = mt.min_row(A);
    int rows_per_process = N / size;  // Разбиение по строкам

    for (index_type row = mt.min_row(A) + rank * rows_per_process; row < mt.min_row(A) + (rank + 1) * rows_per_process; ++row) {
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

        // Обмен данными между процессами
        for (int p = 0; p < size; ++p) {
            if (rank != p) {
                // Отправка и получение строк
                MPI_Send(&A[row], M, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Recv(&A[row], M, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (i = mt.min_row(A); i <= mt.max_row(A); i++) {
            if (i != row)
                add_multiple_row(A, i, row, -mt.element(A, i, lead));
        }
    }
}

void generate_large_matrix(double (&A)[N][M]) {
    random_device rd;                 // Источник случайных чисел
    mt19937 gen(rd());                // Генератор случайных чисел
    uniform_real_distribution<> dis(-1000.0, 1000.0); // Диапазон значений

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = dis(gen);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    alignas(32) double A[N][M];
    if (rank == 0) {
        generate_large_matrix(A);
    }

    MPI_Bcast(A, N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // Рассылка данных всем процессам

    auto start = high_resolution_clock::now();
    to_canonical_rows_form(A, rank, size);
    auto stop = high_resolution_clock::now();

    if (rank == 0) {
        auto duration = duration_cast<nanoseconds>(stop - start);
        std::cout << "Time taken: " << duration.count() << " nanoseconds\n";
    }

    MPI_Finalize();
    return 0;
}
