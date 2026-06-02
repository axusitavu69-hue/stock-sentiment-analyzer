# 创建每日自动学习计划任务（每天18:00收盘后执行）
$action = New-ScheduledTaskAction -Execute "D:\222\daily_run.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At "18:00"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Highest
Register-ScheduledTask -TaskName "StockDailyLearn" -Action $action -Trigger $trigger -Principal $principal -Force
Write-Host "已创建每日18:00自动学习任务" -ForegroundColor Green
Write-Host "查看: taskschd.msc -> 任务计划程序库 -> StockDailyLearn"
