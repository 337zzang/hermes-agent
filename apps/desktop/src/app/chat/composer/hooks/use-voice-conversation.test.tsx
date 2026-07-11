import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// The full-duplex contract: the barge monitor is live across the WHOLE agent
// turn — generation (thinking) and playback (speaking) — so speaking over the
// model interrupts it mid-generation instead of the mic being deaf until TTS
// starts (the Windows report: interruption "never works" because the deaf
// window covered generation, and playback bleed made the old monitor's
// trigger unreachable).

const monitorCalls: BargeMonitorCallbacks[] = []
const stopMonitor = vi.fn()

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: (callbacks: BargeMonitorCallbacks) => {
    monitorCalls.push(callbacks)

    return stopMonitor
  }
}))

const markVoicePlaybackInterrupted = vi.fn()
const stopVoicePlayback = vi.fn()

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: () => markVoicePlaybackInterrupted(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: () => stopVoicePlayback()
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

interface HookProps {
  busy: boolean
}

function renderConversation(overrides: { onInterrupt?: () => void; transcript?: string } = {}) {
  const onInterrupt = overrides.onInterrupt ?? vi.fn()

  // Mirrors the real app: submitting a turn makes the agent busy.
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const onSubmit = vi.fn(async () => {
    onBusyChange.current(true)
  })

  const onStopWord = vi.fn()

  // First transcription is the turn that starts the conversation; subsequent
  // ones are barge captures (the overridable transcript).
  let transcriptions = 0

  const onTranscribeAudio = vi.fn(async () =>
    transcriptions++ === 0 ? 'kick off the task' : (overrides.transcript ?? 'and another thing')
  )

  const hook = renderHook(
    ({ busy }: HookProps) =>
      useVoiceConversation({
        busy,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onInterrupt,
        onStopWord,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      }),
    { initialProps: { busy: false } }
  )

  onBusyChange.current = busy => hook.rerender({ busy })

  return { hook, onInterrupt, onStopWord, onSubmit, onTranscribeAudio }
}

/** Drive the hook into the generation phase (turn submitted, model working). */
async function enterThinking(hook: ReturnType<typeof renderConversation>['hook']) {
  await act(async () => {
    await hook.result.current.start()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('listening'))

  micHandle.stop.mockResolvedValueOnce({
    audio: new Blob(['q'], { type: 'audio/webm' }),
    durationMs: 900,
    heardSpeech: true
  })

  await act(async () => {
    hook.result.current.stopTurn()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
}

describe('useVoiceConversation full-duplex barge-in', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('arms the barge monitor during generation (before any reply audio exists)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)

    await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    // busy=true + thinking → the full-duplex monitor must be live.
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
  })

  it('interrupts the in-flight turn when speech trips mid-generation', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(markVoicePlaybackInterrupted).toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('submits the captured interruption once the interrupt settles (busy clears)', async () => {
    const { hook, onSubmit } = renderConversation({ transcript: 'no, do it differently' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })

    // Interrupt lands → the turn ends → busy flips false.
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('no, do it differently'))
  })

  it('does not interrupt when speech trips during playback (turn already done)', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    // Turn finished; playback phase.
    hook.rerender({ busy: false })

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).not.toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('a spoken stop command in the barge capture ends the conversation instead of submitting', async () => {
    const { hook, onStopWord, onSubmit } = renderConversation({ transcript: 'stop' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['s'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onStopWord).toHaveBeenCalledTimes(1))
    // Only the kickoff turn was submitted — the "stop" capture never was.
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('stop')
  })

  it('re-arms a single monitor per turn (idempotent ensure)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const armed = monitorCalls.length

    // Effect re-runs (busy toggles, status changes) must not open more mics.
    hook.rerender({ busy: true })
    hook.rerender({ busy: true })

    expect(monitorCalls.length).toBe(armed)
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: mocks.getHermesConfigRecord,
  saveHermesConfig: mocks.saveHermesConfig
}))

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: mocks.playSpeechText,
  stopVoicePlayback: mocks.stopVoicePlayback
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: {
      cancel: mocks.handle.cancel,
      start: mocks.handle.start,
      stop: mocks.handle.stop
    },
    level: 0.25
  })
}))

vi.mock('@/lib/realtime-voice-session', () => ({
  RealtimeVoiceSession: class {
    constructor(callbacks: Record<string, (...args: any[]) => void>) {
      mocks.realtimeCallbacks = callbacks
    }

    cancelInput = mocks.realtimeCancelInput
    connect = mocks.realtimeConnect
    disconnect = mocks.realtimeDisconnect
    setMuted = mocks.realtimeSetMuted
  }
}))

import { $voiceInputMode, setVoiceInputMode } from '@/store/voice-prefs'

import { useVoiceConversation } from './use-voice-conversation'

interface HookProps {
  busy: boolean
  enabled: boolean
  mode: 'legacy' | 'realtime'
}

const mountedRoots: Root[] = []

async function waitFor(assertion: () => void) {
  let lastError: unknown

  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      assertion()

      return
    } catch (error) {
      lastError = error
      await act(async () => new Promise(resolve => window.setTimeout(resolve, 0)))
    }
  }

  throw lastError
}

function renderVoiceHook(
  initialProps: HookProps,
  options: Omit<Parameters<typeof useVoiceConversation>[0], keyof HookProps>
) {
  let current: ReturnType<typeof useVoiceConversation>
  let props = initialProps
  const container = document.createElement('div')
  const root = createRoot(container)
  mountedRoots.push(root)

  function Harness(): ReactNode {
    current = useVoiceConversation({ ...options, ...props })

    return null
  }

  act(() => root.render(<Harness />))

  return {
    rerender(next: HookProps) {
      props = next
      act(() => root.render(<Harness />))
    },
    result: {
      get current() {
        return current
      }
    }
  }
}

function setup(initial: HookProps = { busy: false, enabled: false, mode: 'legacy' }) {
  const onSubmit = vi.fn(async () => undefined)
  const onTranscribeAudio = vi.fn(async () => 'legacy transcript')
  const consumePendingResponse = vi.fn()
  const pendingResponse = vi.fn(() => null)

  const hook = renderVoiceHook(initial, {
    consumePendingResponse,
    onSubmit,
    onTranscribeAudio,
    pendingResponse,
    sessionId: 'hermes-session-1'
  })

  return { consumePendingResponse, hook, onSubmit, onTranscribeAudio, pendingResponse }
}

describe('useVoiceConversation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.realtimeCallbacks = null
    mocks.handle.stop.mockResolvedValue({ audio: new Blob(['audio']), heardSpeech: true })
    mocks.getHermesConfigRecord.mockResolvedValue({})
    $voiceInputMode.set('legacy')
  })

  afterEach(() => {
    for (const root of mountedRoots.splice(0)) {
      act(() => root.unmount())
    }
  })

  it('keeps the Legacy recorder -> transcribe -> existing submit path unchanged', async () => {
    const { hook, onSubmit, onTranscribeAudio } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'legacy' })
    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(1))

    const options = mocks.handle.start.mock.calls[0]![0] as { onSilence: () => void }

    await act(async () => {
      options.onSilence()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(onTranscribeAudio).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('legacy transcript')
    expect(mocks.realtimeConnect).not.toHaveBeenCalled()
  })

  it('uses one Hermes submit/tool pipeline per Realtime transcript and makes no Legacy network call', async () => {
    const { hook, onSubmit, onTranscribeAudio } = setup()
    const observedToolEvents: Array<{ name: string; type: string }> = []
    onSubmit.mockImplementationOnce(async () => {
      observedToolEvents.push({ name: 'read_file', type: 'tool.start' })
    })

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeConnect).toHaveBeenCalledWith({ sessionId: 'hermes-session-1' }))

    expect(mocks.handle.start).not.toHaveBeenCalled()
    expect(onTranscribeAudio).not.toHaveBeenCalled()

    await act(async () => {
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-1', text: '첫 번째 질문' })
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-1', text: '첫 번째 질문' })
      await Promise.resolve()
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('첫 번째 질문')
    expect(observedToolEvents).toEqual([{ name: 'read_file', type: 'tool.start' }])
    expect(mocks.realtimeCancelInput).not.toHaveBeenCalled()
  })

  it('queues barge-in while Hermes is busy, stops only TTS, then submits once when ready', async () => {
    const { hook, onSubmit } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeCallbacks).not.toBeNull())
    hook.rerender({ busy: true, enabled: true, mode: 'realtime' })
    expect(mocks.realtimeDisconnect).not.toHaveBeenCalled()

    act(() => {
      mocks.realtimeCallbacks?.onSpeechStarted()
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-barge', text: '새 질문' })
    })

    expect(mocks.stopVoicePlayback).toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit).toHaveBeenCalledWith('새 질문')
  })

  it('surfaces connection errors for explicit retry and cleans the peer on end', async () => {
    const { hook } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeCallbacks).not.toBeNull())

    act(() => mocks.realtimeCallbacks?.onError(new Error('offline')))
    expect(hook.result.current.status).toBe('error')

    await act(async () => hook.result.current.start())
    expect(mocks.realtimeConnect).toHaveBeenCalledTimes(2)

    await act(async () => hook.result.current.end())
    expect(mocks.realtimeDisconnect).toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })
})

describe('Realtime voice preference', () => {
  it('publishes Realtime mode only after the backend feature gate is saved', async () => {
    let releaseSave: (() => void) | undefined

    mocks.getHermesConfigRecord.mockResolvedValue({ voice: { auto_tts: true } })
    mocks.saveHermesConfig.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          releaseSave = resolve
        })
    )

    const saving = setVoiceInputMode('realtime')

    await waitFor(() => expect(mocks.saveHermesConfig).toHaveBeenCalledTimes(1))
    expect($voiceInputMode.get()).toBe('legacy')
    expect(mocks.saveHermesConfig).toHaveBeenCalledWith({
      voice: {
        auto_tts: true,
        input_mode: 'realtime',
        realtime: { enabled: true }
      }
    })

    releaseSave?.()
    await saving
    expect($voiceInputMode.get()).toBe('realtime')
  })
})
