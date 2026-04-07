#include "common/math.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

typedef struct {
    f32 a;
    f32 b;
} LinearRegression;

inline f32 predict(const LinearRegression *model, f32 x) {
    return model->a * x + model->b;
}

inline Table batch_predict(const LinearRegression *model, const Table *X) {
    Table res = init_table(X->rows, 1);
    for (uint i = 0; i < X->rows; i++)
    {
        table_set(&res, i, 0, predict(model, table_get(X, i, 0)));
    }
    return res;
}

void univariate_gradient_descent(LinearRegression *model, const Table *X, const Table *Y, f32 lr, uint max_iters) {
    assert(X->cols == 1 && Y->cols == 1);
    assert(X->rows == Y->rows);
    
    f32 last_loss = INFINITY;
    for (uint iter = 0; iter < max_iters; iter++)
    {
        f32 sum_a = 0.0;
        f32 sum_b = 0.0;
        f32 mse = 0.0;
        for (uint i = 0; i < X->rows; i++)
        {
            f32 x = table_get(X, i, 0);
            f32 y = table_get(Y, i, 0);
            f32 error = predict(model, x) - y;
            sum_a += error * x;
            sum_b += error;
            mse += error * error;
        }
        f32 dA = sum_a / X->rows;
        f32 dB = sum_b / X->rows;
        f32 current_loss = (1.0 / (2.0 * X->rows)) * mse;

        model->a -= lr * dA;
        model->b -= lr * dB;

        if (iter % 100 == 0)
        {
            printf("Iter: %u, Error: %f\n", iter, current_loss);
        }
        if (fabsf(last_loss-current_loss) < 1e-5f || isnan(current_loss) || isinf(current_loss))
        {
            printf("Converged at iter %u\n", iter);
            break;
        }
        last_loss = current_loss;
    }
}

int main() {
    LinearRegression model = {0, 0};
    Table X = init_table(7, 1);
    f32 X_data[] = {1, 2, 3, 4, 5, 6, 7};
    X.data = X_data;
    Table Y = init_table(7, 1);
    f32 Y_data[] = {50, 55, 65, 70, 75, 85, 90};
    Y.data = Y_data;
    univariate_gradient_descent(&model, &X, &Y, 0.01, 300000);
    printf("a: %f, b: %f\n", model.a, model.b);
    return 0;
};