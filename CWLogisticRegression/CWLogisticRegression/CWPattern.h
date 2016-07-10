//
//  CWPattern.h
//  CWLogisticRegression
//
//  Created by Li Chen wei on 2016/7/10.
//  Copyright © 2016年 TWML. All rights reserved.
//

#import <Foundation/Foundation.h>

@protocol CWPatternProtocol <NSObject>

- (NSMutableArray *)feature;
- (double)target;

@end

@interface CWPattern : NSObject <CWPatternProtocol>

@property (nonatomic, strong) NSMutableArray *feature;
@property (nonatomic) double target;

- (instancetype)initWithFeature:(NSArray *)features targetValue:(double)targetValue;

@end
