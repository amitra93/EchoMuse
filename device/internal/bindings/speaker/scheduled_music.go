package speaker

import "sync"

const scheduledMusicRate = 48000

type scheduledMusicChunk struct {
	targetUs int64
	sequence uint32
	pcm      []byte
	offset   int
}

// scheduledMusic renders mono PCM against the device monotonic clock. It is
// deliberately separate from audioStream: that stream is arrival-paced and
// primes before starting, while Sendspin audio must honor presentation times.
type scheduledMusic struct {
	mu           sync.Mutex
	generation   uint32
	active       bool
	ended        bool
	hasSequence  bool
	lastSequence uint32
	lastTargetUs int64
	chunks       []scheduledMusicChunk
	lateSamples  uint64
	underruns    uint64
	corrections  uint64
	started      bool
	startErrorUs int64
	lastErrorUs  int64
}

type scheduledMusicStats struct {
	LateSamples  uint64
	Underruns    uint64
	Corrections  uint64
	StartErrorUs int64
	LastErrorUs  int64
}

func (s *scheduledMusic) start(generation uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if generation == 0 || (s.generation != 0 && generation <= s.generation) {
		return false
	}
	s.generation = generation
	s.active = true
	s.ended = false
	s.hasSequence = false
	s.lastTargetUs = 0
	s.started = false
	s.startErrorUs = 0
	s.lastErrorUs = 0
	s.chunks = nil
	return true
}

func (s *scheduledMusic) push(generation, sequence uint32, targetUs int64, pcm []byte) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.active || generation != s.generation || sequence == 0 && s.hasSequence ||
		(s.hasSequence && sequence <= s.lastSequence) || targetUs < 0 || len(pcm) == 0 || len(pcm)%2 != 0 {
		return false
	}
	if len(s.chunks) > 0 && targetUs < s.lastTargetUs {
		return false
	}
	s.chunks = append(s.chunks, scheduledMusicChunk{
		targetUs: targetUs,
		sequence: sequence,
		pcm:      append([]byte(nil), pcm...),
	})
	s.lastSequence = sequence
	s.hasSequence = true
	s.lastTargetUs = targetUs
	return true
}

func (s *scheduledMusic) clear(generation uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.active || generation != s.generation {
		return false
	}
	s.chunks = nil
	s.hasSequence = false
	s.ended = false
	return true
}

func (s *scheduledMusic) end(generation uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.active || generation != s.generation {
		return false
	}
	s.ended = true
	return true
}

func (s *scheduledMusic) hasStream() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

// render returns one mono period. Silence is returned while waiting for the
// first future timestamp or across a timestamp gap. Late prefixes are skipped
// rather than played late, preserving synchronization at the cost of a rare
// discontinuity that the higher-level telemetry will expose.
func (s *scheduledMusic) render(nowUs int64, samples int) []byte {
	if samples <= 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.active {
		return nil
	}
	result := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		// Round the fractional sample interval instead of truncating every
		// sample. At 48kHz, truncation turns 20.833us into 20us and repeats
		// short PCM chunks at their boundaries.
		target := nowUs + (int64(i)*1_000_000+scheduledMusicRate/2)/scheduledMusicRate
		for len(s.chunks) > 0 {
			chunk := &s.chunks[0]
			if target < chunk.targetUs {
				break
			}
			index := int((target - chunk.targetUs) * scheduledMusicRate / 1_000_000)
			if index < chunk.offset {
				index = chunk.offset
			}
			available := len(chunk.pcm) / 2
			if index >= available {
				s.chunks = s.chunks[1:]
				continue
			}
			if index > chunk.offset {
				s.lateSamples += uint64(index - chunk.offset)
				s.corrections++
			}
			if !s.started {
				s.started = true
				s.startErrorUs = target - chunk.targetUs
			}
			s.lastErrorUs = target - (chunk.targetUs + int64(chunk.offset)*1_000_000/scheduledMusicRate)
			value := chunk.pcm[index*2 : index*2+2]
			copy(result[i*2:i*2+2], value)
			chunk.offset = index + 1
			break
		}
	}
	if s.started && len(s.chunks) == 0 && !s.ended {
		s.underruns++
	}
	for len(s.chunks) > 0 && s.chunks[0].offset >= len(s.chunks[0].pcm)/2 {
		s.chunks = s.chunks[1:]
	}
	if s.ended && len(s.chunks) == 0 {
		s.active = false
	}
	return result
}

func (s *scheduledMusic) stats() scheduledMusicStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return scheduledMusicStats{
		LateSamples: s.lateSamples, Underruns: s.underruns,
		Corrections: s.corrections, StartErrorUs: s.startErrorUs,
		LastErrorUs: s.lastErrorUs,
	}
}
