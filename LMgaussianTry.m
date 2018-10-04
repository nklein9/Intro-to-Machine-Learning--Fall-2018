%plot functions
close all; clear; clc;

a = mvnrnd(25,50,10000,1);
b = mvnrnd(35,50,10000,1);

hist(a,50);
grid on
hold on
%hist(b,50);
%grid on
%hold on