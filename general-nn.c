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

// Matrix typedef
typedef struct 
{
  size_t rows;
  size_t cols;
  // whole matrix in a single 1D array 
  // [ row1, row2, ... rowN]
  float *els;
} Mat;

typedef struct 
{
  size_t lCount;
  Mat *ws;
  Mat *bs;
  Mat *as;
} NN;


float rand_float(void);
Mat matBuild(size_t rows, size_t cols);
void randomizeMat(Mat mat, float mult);
float* matAt(Mat mat, size_t r, size_t c);
Mat matMult(Mat m1, Mat m2);
Mat matAdd(Mat m1, Mat m2);
void sigmoidMat(Mat mat);
void printMat(Mat mat);
void freeMat(Mat mat);
float cost(Mat out, Mat tr);
NN nnBuild(size_t *layerSizes, size_t layerNum);
void nnPrint(NN nn);
void backpropagate(NN nn);



// layerSizes is an array of no. of neurons 
NN nnBuild(size_t *layerSizes, size_t layerNum)
{
  NN nn;
  nn.lCount = layerNum - 1; // subtract the input layer
  
  // Allocate memory to Weights, Biases and Activations
  nn.ws = malloc(sizeof(*nn.ws)*nn.lCount);
  assert(nn.ws != NULL);

  nn.bs = malloc(sizeof(*nn.bs)*nn.lCount);
  assert(nn.bs != NULL);

  nn.as = malloc(sizeof(*nn.as)*(nn.lCount+1));
  assert(nn.as != NULL);
  
  // Build input matrix
  nn.as[0] = matBuild(1, layerSizes[0]);

  // Build maxtrix for each layer
  for (size_t i=1; i < layerNum; i++)
  {
    nn.ws[i-1] = matBuild(nn.as[i-1].cols, layerSizes[i]);
    nn.bs[i-1] = matBuild(1, layerSizes[i]);
    nn.as[i]   = matBuild(1, layerSizes[i]);
  }

  return nn;
}


//Mat forward(NN nn, Mat inp)
//{
//  for (size_t i=2; i < nn.lCount; i++)
//  {
//    
//  }
//}

void nnPrint(NN nn)
{
  printf("\nNeural Network: \n");

  for (size_t i=0; i < nn.lCount; i++)
  {
    printf("\nLayer %zu: \n", i+1);
    printf("Weights: \n");
    printMat(nn.ws[i]);
    printf("Biases: \n");
    printMat(nn.bs[i]);
  }
}


float cost(Mat out, Mat tr)
{
  assert(out.cols == tr.cols);
  assert(out.rows == tr.rows);
  
  float cost = 0;

  for (size_t i=0; i < out.cols*out.rows; i++)
  {
    float diff = (out.els[i] - tr.els[i]);
    cost += diff*diff;
  }

  return cost;
}


float rand_float(void)
{
  return (float) rand() / (float) RAND_MAX;
}


Mat matBuild(size_t rows, size_t cols)
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
  
  Mat res = matBuild(m1.rows, m2.cols);

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

  Mat res = matBuild(rows, cols);

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

  size_t layerCounts[] = {5, 10, 5, 10, 5, 8, 4, 2, 2};
  NN nn = nnBuild(layerCounts, 9);
  nnPrint(nn);

  return 0;
}
