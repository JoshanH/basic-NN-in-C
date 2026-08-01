#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// XOR Gate
float train[][3] = 
{
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};
#define TRAIN_NUM (sizeof(train) / sizeof(train[0]))


typedef struct 
{
  size_t rows;
  size_t cols;
  // whole matrix in a single 1D array 
  // [ row1, row2, ... rowN]
  float *els;
} Mat;


float rand_float(void)
{
  return (float) rand() / (float) RAND_MAX;
}


Mat buildMat(size_t rows, size_t cols)
{
  // allocate matrix list of floats as els
  float *els = malloc(sizeof(float) * rows * cols);
  assert(els);

  for (size_t i=0; i < rows*cols; i++)
  {
    els[i] = 0.0f;
  }

  Mat mat = {rows, cols, els};

  return mat;
}


void randomizeMat(Mat mat, float mult)
{
  for (size_t i = 0; i < mat.cols * mat.rows; i++)
  {
    mat.els[i] = rand_float() * mult;
  }
}


float* matAt(Mat mat, size_t r, size_t c)
{
  return &mat.els[r * mat.cols + c];
}


Mat matMult(Mat m1, Mat m2)
{
  assert(m1.cols == m2.rows);
  
  Mat res = buildMat(m1.rows, m2.cols);

  for (size_t r=0; r < m1.rows; r++)
  {
    for (size_t c=0; c < m2.cols; c++)
    {
      float sum = 0;

      for (size_t k=0; k < m1.cols; k++)
      {
        sum += *matAt(m1, r, k) * *matAt(m2, k, c);
      }

      *matAt(res, r, c) = sum;
    }
  }

  return res;
}


Mat matAdd(Mat m1, Mat m2)
{
  assert((m1.cols == m2.cols) && (m1.rows == m2.rows));

  float rows = m1.rows;
  float cols = m1.cols;

  Mat res = buildMat(rows, cols);

  for (size_t i=0; i < cols * rows; i++)
  {
    res.els[i] = m1.els[i] + m2.els[i];
  }

  return res;
}


void sigmoidMat(Mat mat)
{
  for (size_t i=0; i < mat.rows * mat.cols; i++)
  {
    mat.els[i] = 1 / (1 + expf(-mat.els[i]));
  }
}


void printMat(Mat mat)
{
  for (size_t i=0; i < mat.rows; i++)
  {
    for (size_t j=0; j < mat.cols; j++)
    {
      printf("%f ", *matAt(mat, i, j));
    }
    printf("\n");
  }
}


void freeMat(Mat mat)
{
  free(mat.els);
}


int main()
{
  srand(time(0));
  

  // build weights matrix for 1st layer
  Mat w1 = buildMat(2, 2);
  randomizeMat(w1, 10.0f);
  
  // build bias matrix for 1st layer
  Mat b1 = buildMat(1, 2);
  randomizeMat(b1, 5.0f);

  // build weights matrix for 2nd layer
  Mat w2 = buildMat(2, 1);
  randomizeMat(w2, 10.0f);

  // build bias matrix for 2nd layer
  Mat b2 = buildMat(1, 1);
  randomizeMat(b2, 5.0f);


  printf("\n1st Layer Weights: \n");
  printMat(w1);
  printf("\n1st Layer Bias: \n");
  printMat(b1);

  printf("\n2nd Layer Weights: \n");
  printMat(w2);
  printf("\n2nd Layer Bias \n");
  printMat(b2);


  printf("\n------------------------------------POC-TEST---\n");

  // run a quick little test with the XOR test 
  Mat in = buildMat(1, 2);
  in.els[0] = 1.0f;
  in.els[1] = 1.0f;

  printf("\nInput: \n");
  printMat(in);

  Mat a1 = matAdd(matMult(in, w1), b1);
  sigmoidMat(a1);
  printf("\na1 success\n");
  printMat(a1);

  Mat a2 = matAdd(matMult(a1, w2), b2);
  sigmoidMat(a2);
  printf("a2 success\n");
  printMat(a2);

  printf("\nResult: \n");
  printMat(a2);

  return 0;
}
