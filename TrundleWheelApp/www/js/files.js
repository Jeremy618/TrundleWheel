export default {

  /**
  * Utility function that translates the error code to a string
  */
  errorCodeToString(code) {
    switch (code) {
      case 1:
        return 'NOT_FOUND_ERR'
      case 2:
        return 'SECURITY_ERR'
      case 3:
        return 'ABORT_ERR'
      case 4:
        return 'NOT_READABLE_ERR'
      case 5:
        return 'ENCODING_ERR'
      case 6:
        return 'NO_MODIFICATION_ALLOWED_ERR'
      case 7:
        return 'INVALID_STATE_ERR'
      case 8:
        return 'SYNTAX_ERR'
      case 9:
        return 'INVALID_MODIFICATION_ERR'
      case 10:
        return 'QUOTA_EXCEEDED_ERR'
      case 11:
        return 'TYPE_MISMATCH_ERR'
      case 12:
        return 'PATH_EXISTS_ERR'
      default:
        return 'Unknown Error ' + code
    }
  },

  /**
  * Opens a file.
  * @param {string} filename - filename to be opened
  * @param {boolean} forcecreate - if true the file is created if does not exist
  */
  async openFile(filename, forcecreate) {
    return new Promise((resolve, reject) => {
        let localDir = cordova.file.cacheDirectory
        if (window.cordova.platformId == 'android') localDir = window.cordova.file.externalDataDirectory
        else localDir = window.cordova.file.documentsDirectory

        window.resolveLocalFileSystemURL(localDir, function (dirEntry) {
           console.log('file system opened')
           dirEntry.getFile(filename, { create: forcecreate, exclusive: false }, resolve, reject)
        }, reject)

//        return new Promise((resolve, reject) => {
//          window.resolveLocalFileSystemURL(localDir, function (dir) {
//            dir.getFile(filename, { create: forcecreate }, function (file) {
//              resolve(file);
//            }, function (e) {
//              reject('Cannot get file ' + filename + ', ' + errorCodeToString(e.code))
//            })
//          })
//        })
    })
  },

  async getFilePath(filename) {
    let file = await this.openFile(filename, false)
    return file.nativeURL
  },

  /**
  * Reads a file and delivers the content as an object
  * @param {Object} file - the file to be opened
  */
  async read(file) {
    return new Promise((resolve, reject) => {
      file.file(function (file) {
        var reader = new FileReader()
        reader.onloadend = function () {
          resolve(this.result)
        }
        reader.readAsText(file)
      }, reject)
    })
  },

  /**
  * Deletes a file from the file system.
  * @param {Object} file - the file to be deleted
  */
  async deleteFile(file) {
    return new Promise((resolve, reject) => {
      file.remove(resolve, reject)
    })
  },

  /**
  * Saves txt into the file
  * @param {Object} file - file where to save
  * @param {string} txt - is the text to be saved
  */
  async save(file, txt) {
    if (typeof txt !== 'string') txt = txt.toString()
    return new Promise((resolve, reject) => {
      file.createWriter(function (fileWriter) {
        fileWriter.onwriteend = resolve
        fileWriter.onerror = (err) => {
          console.error(err)
          if (reject) reject()
        }
        fileWriter.write(txt)
      })
    })
  },

  /**
  * Creates a logfile where to append text
  * @param {string} filename - the file name
  */
  async createLog(filename) {
    let file = null
    file = await this.openFile(filename, true)

    return {
      buffer: '',
      writing: false,
      writeBuffer(completed, error) {
        this.writing = true
        let toWrite = this.buffer
        this.buffer = ''

        file.createWriter((fileWriter) => {
          fileWriter.seek(fileWriter.length)
          fileWriter.onerror = error
          fileWriter.onwriteend = () => {
            if (this.buffer.length > 0) {
              // buffer not empty, keep writing
              this.writeBuffer(completed, error)
            } else {
              // buffer empty, completed!
              this.writing = false
              if (completed) completed()
            }
          }
          fileWriter.write(toWrite)
        }, error)
      },
      /**
      * Appends lines to the logger
      * If the logger is busy writing, the promise resolves immediately
      * @param {string} line - the text to be appended, a timestamp and newline are added to it
      */
      async log(line) {
        // add the line to the buffer
        //this.buffer += new Date().toISOString() + ' - ' + line + '\n'
        this.buffer += new Date().getTime() + ' - ' + line + '\n'

        return new Promise((resolve, reject) => {
          // if writing, return immediately
          if (this.writing) resolve()
          else this.writeBuffer(resolve, reject)
        })
      }
    }
  },

  /**
  * Reads a logfile.
  * @param {string} filename - the file name
  */
  async readLog(filename) {
    let file = await this.openFile(filename, true)
    let txt = await this.read(file)
    return txt
  },

  /**
  * Deletes a logfile.
  * @param {string} filename - the file name
  */
  async deleteLog(filename) {
    let file = await this.openFile(filename, false)
    return this.deleteFile(file)
  }

}