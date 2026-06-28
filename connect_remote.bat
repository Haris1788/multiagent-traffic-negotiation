@echo off
REM Connect this local repository to the remote GitHub repo.
REM Run this from the repository root after Git is installed.

cd /d "%~dp0"
if not exist .git (
    git init
)

git remote remove origin 2>nul
if %ERRORLEVEL% neq 0 set ERRORLEVEL=0

git remote add origin https://github.com/Haris1788/multiagent-traffic-negotiation.git
git branch -M main
git add .
git commit -m "Connect local repo to GitHub remote"
git push -u origin main
pause
