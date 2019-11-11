FROM continuumio/miniconda
MAINTAINER hjkuijf
FROM python:3
RUN pip install numpy SimpleITK
RUN pip install deepbrain
ADD python /mrbrains18_csim
