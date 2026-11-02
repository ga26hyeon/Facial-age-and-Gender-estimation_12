[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$ImagePath,
  [Parameter(Mandatory=$true)][string]$LogPath,
  [Parameter(Mandatory=$true)][string]$PhasePath,
  [Parameter(Mandatory=$true)][string]$StopPath,
  [Parameter(Mandatory=$true)][string]$SessionId,
  [string]$Endpoint='http://127.0.0.1:8081/api/analyze'
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Net.Http
$utf8=[System.Text.UTF8Encoding]::new($false)
$client=[System.Net.Http.HttpClient]::new(); $client.Timeout=[TimeSpan]::FromSeconds(30)
function Write-JsonLine([object]$value) {
  [System.IO.File]::AppendAllText($LogPath,(($value|ConvertTo-Json -Compress -Depth 12)+[Environment]::NewLine),$utf8)
}
Write-JsonLine ([ordered]@{type='monitor_started';session=$SessionId;utc=[DateTime]::UtcNow.ToString('o');pid=$PID;endpoint=$Endpoint;image=$ImagePath})
$requestId=0
try {
  while(-not (Test-Path -LiteralPath $StopPath)) {
    $phase='unknown'
    if(Test-Path -LiteralPath $PhasePath){$phase=(Get-Content -LiteralPath $PhasePath -Raw).Trim().TrimStart([char]0xFEFF)}
    $requestId++
    $start=[DateTime]::UtcNow; $watch=[Diagnostics.Stopwatch]::StartNew()
    $status=$null; $revision='unknown'; $revisionPresent=$false; $analysisOk=$false; $faceCount=0; $result=$null; $errorText=$null
    $form=$null; $stream=$null; $response=$null
    try {
      $form=[System.Net.Http.MultipartFormDataContent]::new()
      $stream=[System.IO.File]::OpenRead($ImagePath)
      $content=[System.Net.Http.StreamContent]::new($stream)
      $mediaType = if ([System.IO.Path]::GetExtension($ImagePath) -match '^\.jpe?g$') { 'image/jpeg' } else { 'image/bmp' }
      $content.Headers.ContentType=[System.Net.Http.Headers.MediaTypeHeaderValue]::Parse($mediaType)
      $form.Add($content,'file',[System.IO.Path]::GetFileName($ImagePath))
      $response=$client.PostAsync($Endpoint,$form).GetAwaiter().GetResult()
      $status=[int]$response.StatusCode
      $text=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
      $body=$text|ConvertFrom-Json -ErrorAction Stop
      $properties=@($body.PSObject.Properties.Name)
      $revisionPresent=($properties -contains 'app_revision' -and -not [string]::IsNullOrWhiteSpace([string]$body.app_revision))
      if($revisionPresent){$revision=[string]$body.app_revision}
      if($properties -contains 'faces'){$faces=@($body.faces);$faceCount=$faces.Count;$analysisOk=($status -eq 200 -and $properties -contains 'filename' -and $properties -contains 'notice' -and $revisionPresent)}
      foreach($face in $faces){if(-not $face.box -or [string]::IsNullOrWhiteSpace([string]$face.gender) -or [string]::IsNullOrWhiteSpace([string]$face.age_range) -or $null -eq $face.confidence){$analysisOk=$false}}
      if($faceCount -gt 0){$result=$faces[0]}else{$result='no_face_detected; valid_empty_faces_response'}
      if(-not $analysisOk){$errorText='HTTP response or analysis JSON schema/revision validation failed'}
    } catch { $errorText=$_.Exception.ToString() }
    finally {if($response){$response.Dispose()};if($form){$form.Dispose()};if($stream){$stream.Dispose()}}
    $watch.Stop();$end=[DateTime]::UtcNow
    Write-JsonLine ([ordered]@{type='request';session=$SessionId;phase=$phase;request_id=$requestId;request='POST /api/analyze';start_utc=$start.ToString('o');end_utc=$end.ToString('o');duration_ms=[math]::Round($watch.Elapsed.TotalMilliseconds,1);http_status=$status;analysis_ok=$analysisOk;app_revision=$revision;revision_in_response=$revisionPresent;face_count=$faceCount;representative_result=$result;error=$errorText})
    if(Test-Path -LiteralPath $StopPath){break}
    Start-Sleep -Seconds 1
  }
} finally {
  $client.Dispose()
  Write-JsonLine ([ordered]@{type='monitor_stopped';session=$SessionId;utc=[DateTime]::UtcNow.ToString('o');requests=$requestId})
}
