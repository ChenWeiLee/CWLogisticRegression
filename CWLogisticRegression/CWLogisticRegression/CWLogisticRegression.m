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

- (instancetype)initWithLearnRange:(double)learnRange
{
    self = [self init];
    if (self) {
        _alpha = learnRange;
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
    
    [self updateTheta];
}

- (void)updateTheta
{
    
    for (int index = 0; index < [_thetas count]; index ++) {
        double updateThetaValue = 0.0;
        id<CWPatternProtocol> updatePattern = [_trainingData objectAtIndex:index];
        
        double allSigmaX = 0.0;
        for (int i = 0; i < [_trainingData count]; i ++) {
            // Sum All Cost function
            id<CWPatternProtocol> pattern = [_trainingData objectAtIndex:i];
            
            allSigmaX = allSigmaX + ([self sigmoidWithValue:[self matrixMultiplicationWithMatrix1:_thetas matrix2:[pattern feature]]] - [pattern target]) * [[[updatePattern feature] objectAtIndex:i] doubleValue];
        }
        updateThetaValue = [[_thetas objectAtIndex:index] doubleValue] - _alpha * allSigmaX;

        
        [_thetas replaceObjectAtIndex:index withObject:[NSNumber numberWithDouble:updateThetaValue]];
    }
    
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

@end
