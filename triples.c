#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// multiples of 3 indexed from 0
float train[][2] = {
    {0,0},
    {1,3},
    {2,6},
    {3,9},
    {4,12},
    {5,15},
    {6,18},
    {7,21}
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void)
{
  return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b)
{
  float result = 0; // cost

  for (size_t i=0; i < train_count; i++)
  {
    float x = train[i][0];
    float y = x*w + b;
  
    // cost function uses square difference 
    float d = y - train[i][1];
    result += d*d;
  }
  
  // measures how badly model performed
  // we want this to be close to zero but != 0
  result /= train_count;  
  return result;
}

int main()
{
  // Problem to solve:
  // y = x*w
  // y: output, x: index, w: mystery constant
 
  srand(time(0));
  float w = rand_float() * 10.0f; // initial value
  float b = rand_float() * 5.0f; // bias

  float eps = 1e-3; // 0.001 (arbitrary)
  float rate = 1e-3; // learning rate
  
  for (size_t i=0; i < 400; i++)
  {
    float c = cost(w,b);
    // get the finite difference using formula 
    // (gives approx of a derivative)
    // L = lim_h->0 f(a + h) - f(a) / h
    float dw = (cost(w + eps, b) - c) / eps;
    float db = (cost(w, b + eps) - c) / eps;
    w -= rate * dw; // reduce w by cost differnce * learning rate
    b -= rate * db;

    printf("w: %f bias: %f cost: %f\n", w, b, cost(w, b));
  }
 
  return 0;
}
