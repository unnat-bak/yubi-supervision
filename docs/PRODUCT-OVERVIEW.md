# YUBI Supervision

A live command center for seeing what a room, a recording, or an interview is actually doing.

YUBI Supervision watches a camera or a video and turns it into a running picture: who and what is in frame, how people are moving, and — when you ask for it — how a face departs from its own resting state. The result is a live view an operator can follow in real time, and a report they can hand to someone who was not in the room.

It is built for interviews, recorded conversations, security monitoring, and any review where the useful detail is too fast, too small, or too spread out for a person watching once at normal speed.

---

## What you can point it at

| Source | What happens |
|--------|----------------|
| A webcam on the machine running YUBI | Live, as it happens |
| A video file already on disk | Plays at normal speed, with a clock that matches the file |
| A camera on the network (CCTV / IP) | Live feed, always the latest frame |
| A YouTube link | Resolved and analyzed like any other video |

You choose the source before the session starts. Nothing is uploaded to a browser camera. The feed stays on the machine running YUBI.

## Three ways to watch

Pick a profile on the start screen. Each one turns on the layers that match the job and leaves the rest quiet.

**General.** Objects, body, face, and hands, plus a plain-language read of the scene. The default for "show me what is in this frame."

**Interview.** The behavioral session. Expression tracking, body language, and a written analysis at the end. A recorded file plays once and stops when the file ends, so the session lines up with the interview instead of looping. This is the profile for a conversation you want to study.

**Security.** People and objects, a durable identity on each tracked person, and alerts when something on your watchlist appears.

You can still turn individual layers on and off during a session: objects, masks, skeleton, face, hands, joint labels, identity, and expressions.

---

## The live view

The screen is a single command center.

- **The feed** is the annotated video: boxes, labels, skeletons, and expression markup, only for the layers you have on.
- **The top bar** shows the time, the session, how long it has been running, which source you are on, which profile is active, and — for a file — where you are in the recording. A small trace shows how smoothly the pipeline is keeping up.
- **The left side** is the object list, the layer controls, and a live event feed: system notes, scene updates, objects, notable expressions, insights, and alerts, in order.
- **The right side** is where interpretation lives: a scene summary, and, in an interview, the expression panel.

From the bottom bar you can start, stop, snapshot a frame, or record the annotated video.

---

## Expression reading

This is the part of YUBI that notices what a careful viewer still misses, because it is brief, because it is subtle, or because speech is covering it.

### It learns the person in front of it

A resting frown is not anger, and a naturally lively face is not excitement. In the first moments with each person, YUBI learns that face at rest and measures everything after against that person, not against a generic face. Two people in the same interview each get their own baseline. What counts is the change.

### Fleeting and held

Some expressions are on the face and gone before a viewer can name them. Others stay long enough to be obvious. YUBI keeps both, and marks which is which. The short ones are the point of the product: the flash of tightening around the eyes, the one-sided pull at the mouth, the smile that arrives and is put away. They are logged with a time, a duration, and the part of the face that moved, so you can go back to the moment instead of trusting a recollection.

### Feelings are named carefully

YUBI describes affect in ordinary words: joy, surprise, fear, anger, disgust, sadness, contempt, or simply neutral when the movement does not earn a stronger name.

A name is earned by the movement happening now. A lip movement during an ongoing frown is described as that lip movement. It is not quietly relabeled with the frown's feeling. A real smile that reaches the eyes is distinguished from a polite smile that stays at the mouth. A pull on one side of the mouth is called out on its own, because that asymmetry is one of the clearest contempt signals a face produces.

### Talking is not an emotion

Speech moves the mouth constantly, and people lift and lower their brows to punctuate a sentence. A system that treats all of that as feeling will fill a report with false anger and false surprise.

YUBI treats speech as speech. While someone is talking, ordinary mouth movement is set aside, and a brief brow raise or brow lower that lands with the words is labeled as emphasis — a conversational beat — rather than as an emotion. The feeling that remains is the feeling that is actually on the face. Emphasis stays on the timeline, so you can see how someone stresses a point, without it being counted as affect.

### Signals a viewer does not keep score of

Alongside the expression itself, YUBI keeps a quiet account of behavior that is real and easy to under-count by eye:

- **How often they blink**, including bursts of several blinks close together and moments when the eyes stay closed a little too long. A rate that runs well above a person's own pace is a sign of load, even when the words sound steady.
- **Where they look**, including a sustained look away from the camera.
- **Nods and shakes**, the small vertical and horizontal movements people use to agree, encourage, or refuse.
- **A hand to the face**, the self-touch people use to settle themselves.
- **A pulse estimate**, offered only when the signal from the skin is strong enough to trust, and always marked as experimental.

Each of these is a count and a rate over the time that person was actually on screen, not a vibe.

### Patterns, not just single movements

Single movements are the evidence. The patterns are what an interviewer wants named:

- A smile of enjoyment, against a smile that is only polite.
- A one-sided smirk.
- Lips pressed or drawn in, the small controls people use when they are holding something back or steadying themselves.
- A negative movement covered by a smile.
- A negative flash that shows through a smile that was already there.
- An expression that starts and is cut off.
- A frown of concentration, a chin raised in doubt, a brow that worries.

Every pattern is tied to a moment on the clock. None of them is a verdict.

### What you see while it runs

The expression panel stays with the person on screen:

- The current read for each person, and whether their personal baseline is still forming.
- How positive or negative the moment is, and which feelings are present right now.
- A timeline of activation across the session, with the short flashes marked so they stand out from the longer expressions.
- Which regions of the face are active: brow, eyes, nose, mouth.
- The running physiology: blinks, gaze, head, pulse, whether they are speaking, whether a hand has come to the face.
- The list of episodes so far, each with its time, how long it lasted, and the pattern attached to it.

The event feed on the left records the moments worth noticing as they happen, so the operator is not only watching the face.

---

## A second opinion, only when something happened

YUBI v3.0 is the language layer. It does not narrate every frame.

It writes a short description of the scene, and it checks objects the tracker is unsure about so a label can stick to that person or thing instead of flickering. In an interview, it also reads the moments the expression system has already flagged — a brief expression, a covered feeling, a strong negative turn — and offers a hedged interpretation in plain language, with the measured moment it is based on. A running summary builds as the session goes, so you can glance at where the conversation has been without scrubbing back.

The interpretation is an opinion about the evidence. The measurements underneath it are kept either way. If the language layer is unavailable, or if a session reaches the spend limit you set, the measurements continue and the report is still complete.

You can see the spend as the session runs: how many interpretations have been made, and how much of the budget is left. The system is built to stay quiet when the scene is quiet, and to speak when something changed.

---

## Objects, people, and the room

- **Objects** are boxed, named, and followed, so a person or an item keeps an identity as they move.
- **Masks** can outline each detected thing when you want the shape, not only the box.
- **Body, face, and hands** can be drawn as a light skeleton, with optional labels on the major joints.
- **Identity** can attach a name, a number, or a short description to a tracked person and keep it there, which is what the Security profile is for.
- **Scene summary** says, in a sentence or two, what the frame is. It updates when the scene changes, not on a timer that ignores whether anything happened.
- **Alerts** fire when a class you listed shows up — a person, a phone, whatever you name — as a banner on screen and, if you want, a message to a webhook you control.

---

## When the session ends

Stopping the session, or letting an interview file play to its end, opens a short close-out.

**The measured log.** A document of the session: what the source was, which profile ran, how long it lasted, and the event-by-event chronology. For an interview it leads with a behavioral analysis written from the measurements:

- One row per person: how long they were observed, how many brief and held expressions, blinks per minute, time looking away, nods and shakes, self-touch, the experimental pulse when it was available, and how their affect was distributed over time.
- The moments worth a second look, each with a time, the person, what was seen, and how long it lasted.
- A tally of the patterns, each with a plain description of what that pattern can indicate.

**The timeline export.** Every episode, in a file you can open in a spreadsheet or hand to another tool: who, when, how long, what it was, and the pattern attached to it. This is the audit trail. The prose report is the reading; the export is the evidence.

**The intelligence brief.** Optional. YUBI v3.0 reads the measured record and writes a shorter brief, then checks that brief back against the evidence so a claim that the measurements do not support does not survive into the document. If that pass cannot run, you still leave with the measured log.

---

## What this is, and what it is not

YUBI reports what the face and the body did, relative to that person's own baseline, at a specific time.

Those signals are observational. A blink rate, a covered smile, a look away, or a flash of contempt-like movement is a reason to rewatch a moment. It is not evidence of deception, intent, or a clinical state, and the product will not tell you that it is. A trained interviewer uses it to decide where to look again. The system does not replace that judgment, and it should not be used as one.

A few practical limits, stated plainly:

- A hard cut in a recording can cause the same person to be treated as someone new on the other side of the cut.
- Speech awareness is visual. It does not listen to the audio track.
- The pulse figure is experimental and is withheld when it is not confident.
- Very compressed video can make small movements, especially blinks, harder to count cleanly.

---

## A session, end to end

1. Choose the source and the profile. For a recorded interview, choose Interview.
2. Start. For a file, the clock on screen matches the recording.
3. Watch the feed, the expression panel, and the event list. Notable moments appear as they happen. The language layer adds a short interpretation when a moment earns one.
4. Snapshot or record anything you want to keep as video.
5. When the file ends, or when you stop, download the measured log, export the full timeline, or request the intelligence brief.

What you take away is a timed account of behavior that a single viewing does not reliably produce: the flashes under half a second, the smile that never reached the eyes, the emphasis that was only emphasis, and the slow shift in how a person looked across the length of the conversation.
