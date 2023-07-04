# docker file for building the image for nextjs frontend app
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps

# below line is needed for the nextjs app to run in alpine linux container
RUN apk add --no-cache libc6-compat

# Create app directory
WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./

RUN \
    if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then yarn global add pnpm && pnpm i --frozen-lockfile; \
    else echo "Lockfile not found." && exit 1; \
    fi

# Rebuild the source code only when needed
FROM deps AS builder

# Copy all the files from the project’s root to the working directory
COPY . .

# Build the project
RUN yarn build

# Expose port 3000
EXPOSE 3000

# Run the app in production mode
CMD ["yarn", "start"]
