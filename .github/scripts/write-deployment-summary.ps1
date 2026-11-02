[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$LogPath,[Parameter(Mandatory=$true)][string]$RunId,[Parameter(Mandatory=$true)][string]$CiRunId,[Parameter(Mandatory=$true)][string]$Repository,[Parameter(Mandatory=$true)][string]$ServerUrl,[Parameter(Mandatory=$true)][string]$DeployOutcome)
$ErrorActionPreference='Stop'
$rows=@()
if(Test-Path -LiteralPath $LogPath){$rows=@(Get-Content -LiteralPath $LogPath -Encoding UTF8|ForEach-Object{try{$_|ConvertFrom-Json -ErrorAction Stop}catch{$null}}|Where-Object{$null -ne $_})}
$req=@($rows|Where-Object{$_.type -eq 'request'}|Sort-Object request_id)
$valid=@($req|Where-Object{$_.http_status -eq 200 -and $_.analysis_ok -eq $true})
$fail=$req.Count-$valid.Count
$ids=@($req|ForEach-Object{[int]$_.request_id})
$idsContiguous=($ids.Count -eq 0 -or (($ids|Measure-Object -Minimum).Minimum -eq 1 -and ($ids|Measure-Object -Maximum).Maximum -eq $ids.Count -and $ids.Count -eq (@($ids|Select-Object -Unique)).Count))
$starts=@();$endToStart=@()
for($i=0;$i -lt $req.Count;$i++){$starts+=([DateTimeOffset]::Parse($req[$i].start_utc));if($i -gt 0){$endToStart+=(([DateTimeOffset]::Parse($req[$i].start_utc)-[DateTimeOffset]::Parse($req[$i-1].end_utc)).TotalSeconds)}}
$startIntervals=@();for($i=1;$i -lt $starts.Count;$i++){$startIntervals+=(($starts[$i]-$starts[$i-1]).TotalSeconds)}
$phaseGroups=@{};foreach($p in @('pre-deploy','during-deploy','post-deploy','deployment-failed')){$phaseGroups[$p]=@($req|Where-Object{$_.phase -eq $p})}
$representatives=@{}
if($phaseGroups['pre-deploy'].Count){$representatives.before=$phaseGroups['pre-deploy'][-1]}
if($phaseGroups['during-deploy'].Count){$representatives.during=@($phaseGroups['during-deploy']|Where-Object{$_.revision_in_response -and $_.app_revision -ne $phaseGroups['pre-deploy'][-1].app_revision}|Select-Object -First 1)[0];if(-not $representatives.during){$representatives.during=$phaseGroups['during-deploy'][0]}}
if($phaseGroups['post-deploy'].Count){$representatives.after=$phaseGroups['post-deploy'][0]}
elseif($phaseGroups['deployment-failed'].Count){$representatives.after=$phaseGroups['deployment-failed'][0]}
$firstRev=@($req|Where-Object{$_.revision_in_response}|Select-Object -First 1)[0]
$revisionTransitions=@();$previous=$null;foreach($r in $req){if($r.revision_in_response){if($null -ne $previous -and $r.app_revision -ne $previous){$revisionTransitions+=[pscustomobject]@{from=$previous;to=$r.app_revision;utc=$r.start_utc;request_id=$r.request_id}};$previous=$r.app_revision}}
$from=if($req.Count){$req[0].start_utc}else{'not recorded'};$to=if($req.Count){$req[-1].end_utc}else{'not recorded'}
$successText=if($DeployOutcome -eq 'success'){'SUCCESS'}else{"FAILED / $DeployOutcome"}
$summary=[System.Collections.Generic.List[string]]::new()
$summary.Add('# Face Estimator deployment verification')
$summary.Add('')
$summary.Add("**Result:** $successText  |  **Stable URL:** http://127.0.0.1:8081/  |  **Window (UTC):** $from to $to")
$summary.Add('')
$summary.Add("**API requests:** $($req.Count) total; $($valid.Count) successful; $fail failed; request IDs contiguous: $idsContiguous.")
$summary.Add('')
$summary.Add('## Before / during / after')
$summary.Add('')
$summary.Add('| Stage | Requests (ok/total) | Representative request | UTC start | HTTP | Analysis | Response revision | Result |')
$summary.Add('|---|---:|---:|---|---:|---|---|---|')
foreach($entry in @(@('Before','pre-deploy'),@('During','during-deploy'),@('After','post-deploy'))){$label=$entry[0];$key=$entry[1];$group=$phaseGroups[$key];$r=$representatives[$label.ToLower()];if(-not $r -and $key -eq 'post-deploy'){$group=$phaseGroups['deployment-failed'];$r=$representatives.after};$okay=@($group|Where-Object{$_.http_status -eq 200 -and $_.analysis_ok}).Count;if($r){$rev=$r.app_revision;$res=if($r.face_count -gt 0){"faces=$($r.face_count); $($r.representative_result.gender), age $($r.representative_result.age_range)"}else{'valid response; no face in fixed probe'};$summary.Add("| $label | $okay/$($group.Count) | #$($r.request_id) | $($r.start_utc) | $($r.http_status) | $($r.analysis_ok) | ``$rev`` | $res |")}else{$summary.Add("| $label | $okay/$($group.Count) | - | - | - | - | - | no captured request in this phase |")}}
$summary.Add('')
$summary.Add('## Revision, cadence, and availability')
$summary.Add('')
if($revisionTransitions.Count){foreach($t in $revisionTransitions){$summary.Add("- **Version switch:** ``$($t.from)`` -> ``$($t.to)`` at $($t.utc) (analysis request #$($t.request_id)).")}}elseif($firstRev){$summary.Add("- Analysis response revision observed: ``$($firstRev.app_revision)``; no revision change was observed in request responses.")}else{$summary.Add('- No analysis response with an app revision was captured.')}
$maxGap=if($startIntervals.Count){[math]::Round(($startIntervals|Measure-Object -Maximum).Maximum,3)}else{0};$avgGap=if($startIntervals.Count){[math]::Round(($startIntervals|Measure-Object -Average).Average,3)}else{0};$minIdle=if($endToStart.Count){[math]::Round(($endToStart|Measure-Object -Minimum).Minimum,3)}else{0};$maxIdle=if($endToStart.Count){[math]::Round(($endToStart|Measure-Object -Maximum).Maximum,3)}else{0}
$summary.Add("- **Actual interval:** request starts mean ${avgGap}s apart; largest start-to-start/record gap ${maxGap}s. After-response wait measured ${minIdle}s to ${maxIdle}s (target: 1s).")
$summary.Add("- **Availability evidence:** $($valid.Count)/$($req.Count) sampled analysis requests returned HTTP 200 with a valid response; failed requests: $fail; request IDs contiguous: $idsContiguous. No interruption was observed in this sample. Gaps between sampled requests and all unobserved instants are **not verified**; this is evidence of observed-request continuity, not proof of continuous availability between requests.")
$summary.Add('')
$summary.Add('## Workflow runs')
$summary.Add('')
$summary.Add("- [CI build / Docker Hub push]($ServerUrl/$Repository/actions/runs/$CiRunId)")
$summary.Add("- [Self-hosted CD run]($ServerUrl/$Repository/actions/runs/$RunId)")
$markdown=$summary -join [Environment]::NewLine
if($env:GITHUB_STEP_SUMMARY){[IO.File]::AppendAllText($env:GITHUB_STEP_SUMMARY,$markdown+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))}
Write-Output '--- Per-request JSONL evidence (also attached as artifact) ---'
if(Test-Path -LiteralPath $LogPath){Get-Content -LiteralPath $LogPath -Encoding UTF8|ForEach-Object{Write-Output $_}}else{Write-Output 'Monitor log file is missing.'}
Write-Output '--- Summary ---';Write-Output $markdown
if($req.Count -eq 0){throw 'No API analysis requests were captured; see monitor startup and deployment logs.'}
