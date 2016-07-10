//
//  CWLogisticRegression.h
//  CWLogisticRegression
//
//  Created by Li Chen wei on 2016/7/10.
//  Copyright © 2016年 TWML. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "CWPattern.h"

@interface CWLogisticRegression : NSObject

- (instancetype)initWithLearnRange:(double)learnRange iteration:(int)iteration;
- (void)trainingWithPatterns:(NSMutableArray <id<CWPatternProtocol>>*)patterns;

- (double)outputWithData:(id<CWPatternProtocol>)data;

@end
