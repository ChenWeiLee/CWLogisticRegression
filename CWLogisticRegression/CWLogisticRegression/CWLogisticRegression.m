//
//  CWLogisticRegression.m
//  CWLogisticRegression
//
//  Created by Li Chen wei on 2016/7/10.
//  Copyright © 2016年 TWML. All rights reserved.
//

#import "CWLogisticRegression.h"

@interface CWLogisticRegression ()

@property (nonatomic, strong) NSMutableArray *thetas;
@property (nonatomic) double alpha;
@property (nonatomic) int iteration;

@property (nonatomic, strong) NSMutableArray <id <CWPatternProtocol>>*trainingData;

@end

@implementation CWLogisticRegression

- (instancetype)init
{
    self = [super init];
    if (self) {
        _thetas = [NSMutableArray new];
        _trainingData = [NSMutableArray new];
    }
    return self;
}

- (instancetype)initWithLearnRange:(double)learnRange iteration:(int)iteration
{
    self = [self init];
    if (self) {
        _alpha = learnRange;
        _iteration = iteration;
    }
    return self;
}

#pragma mark - Output

- (double)outputWithData:(id<CWPatternProtocol>)data
{
    if ([[data feature] count] != [_thetas count]) {
        return NSNotFound;
    }
    
    return [self sigmoidWithValue:[self matrixMultiplicationWithMatrix1:_thetas matrix2:[data feature]]];
}

#pragma mark - Train
//Andrew
//https://www.coursera.org/learn/machine-learning/lecture/MtEaZ/simplified-cost-function-and-gradient-descent
- (void)trainingWithPatterns:(NSMutableArray <id<CWPatternProtocol>>*)patterns
{
    
    int index = 0;
    do {
        [_thetas addObject:[NSNumber numberWithDouble:0.0]];
        index ++;
    } while (index < [[[patterns objectAtIndex:0] feature] count]);
    
    _trainingData = [patterns mutableCopy];
    
    for (int iter = 0; iter < _iteration; iter ++) {
        [self updateTheta];
        
        if ([self costValue] <= 0.0001) { //檢查閥值
            return;
        }
    }
}

- (void)updateTheta
{
    NSMutableArray *newTheta = [NSMutableArray new];
    
    for (int index = 0; index < [_thetas count]; index ++) {
        double updateThetaValue = 0.0;
        
        double allSigmaX = 0.0;
        for (int i = 0; i < [_trainingData count]; i ++) {
            // Sum All Cost function
            id<CWPatternProtocol> pattern = [_trainingData objectAtIndex:i];
            
            allSigmaX = allSigmaX + ([self sigmoidWithValue:[self matrixMultiplicationWithMatrix1:_thetas matrix2:[pattern feature]]] - [pattern target]) * [[[pattern feature] objectAtIndex:index] doubleValue];
        }
        updateThetaValue = [[_thetas objectAtIndex:index] doubleValue] - _alpha * allSigmaX;

        [newTheta addObject:[NSNumber numberWithDouble:updateThetaValue]];
    }
    _thetas = [newTheta mutableCopy];
}

- (double)matrixMultiplicationWithMatrix1:(NSMutableArray *)matrix1 matrix2:(NSMutableArray *)matrix2
{
    if ([matrix1 count] != [matrix2 count]) {
        return NSNotFound;
    }
    
    double result = 0.0;
    for (int index = 0; index < [matrix1 count]; index++) {
        result = result + [[matrix1 objectAtIndex:index] doubleValue] * [[matrix2 objectAtIndex:index] doubleValue];
    }
    
    return result;
}

- (double)sigmoidWithValue:(double)value
{
    return 1/(1 + exp(value));
}

- (double)costValue
{
    double costValue = 0.0;

    //Cost function
    for (int i = 0; i < [_trainingData count]; i ++) {
        id<CWPatternProtocol> pattern = [_trainingData objectAtIndex:i];
        
        costValue = costValue + pow(([self sigmoidWithValue:[self matrixMultiplicationWithMatrix1:_thetas matrix2:[pattern feature]]] - [pattern target]), 2);
    }
    
    return costValue/(2 * [_trainingData count]);
}

@end
