# Multi-Modal LLM Project

## Objective
Create a multi-modal Large Language Model (LLM) that can process:
1. Text
2. Image
3. Audio

The output will be text-based.

## Implementation Guidelines

### 1. Training

#### 1.1 Image Processing
- Use the original Instruct 150k dataset
- Utilize CLIP for image embeddings:
  - Option A: Run in real-time (requires more GPU power)
  - Option B: Preprocess and store embeddings
- Add a projection layer:
  - Convert CLIP embeddings to a format compatible with your Phi Model
  - Do not apply QLoRa to this layer
- Implement an adapter:
  - Train using QLoRa on the Instruct 150k dataset

#### 1.2 Audio Processing
- Use Whisper for Automatic Speech Recognition (ASR)
- Add a projection layer for Whisper output (text-only)
- Note: This part should not require training, but needs proper integration with your model

#### 1.3 Text Processing
- Assume this component is already implemented

### 2. Deployment
- Create a ChatGPT-like interface that allows:
  - Text input
  - Image upload
  - Audio upload (live recording or file)

## Submission Requirements

1. HuggingFace Spaces App Link
2. GitHub repository link containing all project files
3. Detailed README file including:
   - All logs
   - Full project description
   - Potential improvements

## Deadlines

- First Submission: September 21-28
  - Only this submission required for Course Completion Certificate
- Late Submission: October 30
  - 60% overall score required for Course Completion Certificate

## Additional Resources

- Understanding LLaVA: Large Language and Vision Assistant
  - https://voxel51.com/blog/understanding-llava-large-language-and-vision-assistant/
- LLaVA: Large Language and Vision Assistant
  - https://arxiv.org/pdf/2310.03744

## Utility Commands

### File Transfer (Local to EC2)

```bash
scp -i ~/.ssh/aws-ssh1.pem -r multimodal-system ubuntu@ec2-65-1-95-76.ap-south-1.compute.amazonaws.com:/home/ubuntu
```

### File Transfer (EC2 to Local)

```bash
scp -i ~/.ssh/aws-ssh1.pem -r ubuntu@ec2-65-1-95-76.ap-south-1.compute.amazonaws.com:/home/ubuntu/multimodal-system .
```

### Modify Git Config

```bash
git config --global user.name "aakashvardhan"
git config --global user.email "amadabushi@hawk.iit.edu"
```